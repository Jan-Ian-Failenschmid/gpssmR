// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::depends(RcppProgress)]]

#include <RcppArmadillo.h>
#include "imc_gp_class.h"
#include "imc_gp_helper.h"
#include "linear_algebra.h"
#include "pdfs.h"

// Update functions
void imc_gp::update_train_data(const arma::mat &training_data_inp,
                               const arma::mat &outcome_data_inp)
{
    train_dat = training_data_inp;
    n_training = train_dat.n_cols;

    outcome_dat = outcome_data_inp;
    n_outcome = outcome_dat.n_cols;

    dim_inp = train_dat.n_rows;
    dim_out = outcome_dat.n_rows;

    train_mu = mu(train_dat);
    arma::mat train_k = kernel(train_dat);
    // train_k.diag() += 1;
    jitter_mat(train_k, delta);

    train_k_chol = arma::chol(train_k, "lower");
}

void imc_gp::append_train_data(const arma::mat &training_data_inp,
                               const arma::mat &outcome_data_inp)
{
    arma::mat mu_new_train = mu(training_data_inp);
    arma::mat k_lower_block = kernel(training_data_inp, train_dat);
    arma::mat k_lower_diag = kernel(training_data_inp);
    jitter_mat(k_lower_diag, delta);

    train_dat = join_rows(train_dat, training_data_inp);
    n_training += training_data_inp.n_cols;

    outcome_dat = join_rows(outcome_dat, outcome_data_inp);
    n_outcome += outcome_data_inp.n_cols;

    train_mu = join_rows(train_mu, mu_new_train);
    add_cholesky_lower(train_k_chol, k_lower_block, k_lower_diag);
}

void imc_gp::update_test_data(const arma::mat &test_data_inp)
{
    test_dat = test_data_inp;
    n_test = test_dat.n_cols;

    test_mu = mu(test_dat);
    arma::mat test_k = kernel(test_dat);
    jitter_mat(test_k, delta);
    test_k_chol = chol(test_k, "lower");
}

void imc_gp::append_test_data(const arma::mat &test_data_inp)
{
    arma::mat mu_new_test = mu(test_data_inp);
    arma::mat test_lower_block = kernel(test_data_inp, test_dat);
    arma::mat test_lower_diagonal = kernel(test_data_inp);
    jitter_mat(test_lower_diagonal, delta);

    test_dat = join_rows(test_dat, test_data_inp);
    n_test += test_data_inp.n_cols;

    test_mu = join_rows(test_mu, mu_new_test);
    add_cholesky_lower(test_k_chol, test_lower_block, test_lower_diagonal);
}

void imc_gp::reset_test_data()
{
    test_dat = test_dat.col(0);
    n_test = 1;

    test_mu.resize(dim_out, 1);
    test_k_chol.resize(1, 1);
}

void imc_gp::set_train_y_cov(const arma::mat &y_cov_)
{
    train_y_cov_chol = arma::chol(y_cov_, "lower");
}

void imc_gp::set_train_y_cov_I()
{
    set_train_y_cov(identity(n_outcome));
}

void imc_gp::append_train_y_cov_I()
{
    arma::uword n_old = train_y_cov_chol.n_cols;
    arma::uword n_new = n_outcome - n_old;
    arma::mat y_lower_block(n_new, n_old, arma::fill::zeros);
    arma::mat y_lower_diag = identity(n_new);

    append_train_y_cov(y_lower_block, y_lower_diag);
}

void imc_gp::append_train_y_cov(
    const arma::mat &y_lower_block,
    const arma::mat &y_lower_diag)
{
    add_cholesky_lower(train_y_cov_chol, y_lower_block,
                       y_lower_diag);
}

void imc_gp::set_test_y_cov_I()
{
    set_test_y_cov(identity(n_test));
}

void imc_gp::set_test_y_cov(const arma::mat &y_cov_)
{
    test_y_cov_chol = arma::chol(y_cov_, "lower");
}

void imc_gp::append_test_y_cov_I()
{
    arma::uword n_old = test_y_cov_chol.n_cols;
    arma::uword n_new = n_test - n_old;
    arma::mat y_lower_block(n_new, n_old, arma::fill::zeros);
    arma::mat y_lower_diag = identity(n_new);

    append_test_y_cov(y_lower_block, y_lower_diag);
}

void imc_gp::append_test_y_cov(const arma::mat &y_lower_block,
                               const arma::mat &y_lower_diag)
{
    add_cholesky_lower(test_y_cov_chol, y_lower_block,
                       y_lower_diag);
}

void imc_gp::reset_test_y_cov()
{
    test_y_cov_chol.resize(1, 1);
}

void imc_gp::update_hyperparameters(const double &alpha_inp,
                                    const double &rho_inp)
{
    alpha = alpha_inp;
    rho = rho_inp;

    arma::mat train_k = kernel(train_dat);
    jitter_mat(train_k, delta);
    train_k_chol = chol(train_k, "lower");

    arma::mat test_k = kernel(test_dat);
    jitter_mat(test_k, delta);
    test_k_chol = chol(test_k, "lower");
}

void imc_gp::update_sigma(const arma::mat &sigma_inp)
{
    sigma = sigma_inp;
    sigma_chol = arma::chol(sigma, "lower");
}

arma::mat imc_gp::kernel(const arma::mat &x1, const arma::mat &x2)
{
    return gp_covariance_multi(x1, x2, rho, alpha);
}

arma::mat imc_gp::kernel(const arma::mat &x1)
{
    return gp_covariance_multi(x1, rho, alpha);
}

arma::mat imc_gp::mu(const arma::mat &x)
{
    return arma::zeros(dim_out, x.n_cols);
}

double imc_gp::marginal_log_likelihood()
{
    // Log liklihood of outcomes at training data after marginalizing out the GP

    marg_log_lik = logdmatnorm(outcome_dat, train_mu,
                               sigma_chol, get_marginal_train_cov_chol());
    return (marg_log_lik);
}

double imc_gp::test_marginal_log_likelihood(const arma::mat &test_outcome)
{
    compute_predictive(true);

    // Log liklihood of outcomes at training data after marginalizing out the GP
    marg_log_lik = logdmatnorm(test_outcome, pred_mean,
                               sigma_chol,
                               pred_col_cov_chol);
    return (marg_log_lik);
}

double imc_gp::prior_log_likelihood()
{
    log_lik = logdmatnorm(gp_val, train_mu, sigma_chol, train_k_chol);
    return (log_lik);
}

arma::mat imc_gp::get_marginal_train_cov_chol()
{
    return chol_of_sum(train_k_chol, train_y_cov_chol);
}

void imc_gp::compute_predictive(const bool &marginal)
{
    // Calculate K^-1*K^T using L-T*L*-1*K^T
    // Solves throws warnings, use fast to avoid warnings, since K + I is PD
    test_train_k = kernel(test_dat, train_dat);
    arma::mat L = get_marginal_train_cov_chol();
    arma::mat L_inv_k_T = chol_left_solve(L, test_train_k.t());
    arma::mat K_inv_k_T = arma::solve(arma::trimatu(L.t()),
                                      L_inv_k_T,
                                      arma::solve_opts::fast);

    arma::mat test_cov_chol = test_k_chol;
    arma::mat test_cov = test_k_chol * test_k_chol.t();
    if (marginal)
    {
        test_cov += test_y_cov_chol * test_y_cov_chol.t();
    }
    pred_mean = test_mu + (outcome_dat - train_mu) * K_inv_k_T;
    pred_col_cov_chol = chol(test_cov - L_inv_k_T.t() * L_inv_k_T, "lower");
}

arma::mat imc_gp::make_marginal_predictions()
{
    compute_predictive(true);
    return pred_mean +
           sigma_chol * arma::randn(dim_out, n_test) * pred_col_cov_chol.t();
}

arma::mat imc_gp::make_test_predictions()
{
    compute_predictive(false);
    gp_val = pred_mean +
             sigma_chol * arma::randn(dim_out, n_test) * pred_col_cov_chol.t();
    return gp_val;
}
