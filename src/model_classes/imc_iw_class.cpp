// [[Rcpp::depends(RcppArmadillo)]]

#include <RcppArmadillo.h>
#include "imc_iw_class.h"
#include "imc_gp_class.h"
#include "linear_algebra.h"
#include "pdfs.h"

imc_iw::imc_iw(
    const arma::vec &hyperparameters_inp,
    const arma::mat &covar_mean_prior_inp,
    const arma::mat &covar_col_cov_prior_inp,
    const int &cov_df_inp,
    const arma::mat &cov_scale_inp)
    : cov_df(cov_df_inp),
      cov_scale(cov_scale_inp),
      covar_mean_prior(covar_mean_prior_inp),
      covar_col_cov_prior(covar_col_cov_prior_inp),
      multiv_gp(),
      post_cov_scale(),
      post_cov_scale_chol(),
      d_outcome(),
      d_predictor(),
      n_time()
{
    des_covar_mat.set_size(covar_mean_prior.n_rows, covar_mean_prior.n_cols);
    covar_col_cov_post.set_size(covar_col_cov_prior.n_rows,
                                covar_col_cov_prior.n_cols);
    covar_col_cov_prior_inv = arma::inv_sympd(covar_col_cov_prior);
    covar_col_cov_prior_chol = chol(covar_col_cov_prior, "lower");
    // covar_col_cov_post_chol = chol(covar_col_cov_post, "lower");
    cov_scale_chol = chol(cov_scale, "lower");
    update_hyperparameters(hyperparameters_inp[0], hyperparameters_inp[1]);
}

void imc_iw::update_data(
    const arma::mat &outcome_inp,
    const arma::mat &predictor_inp,
    const arma::mat &covariate_inp)
{
    predictor = predictor_inp;
    outcome = outcome_inp;
    covariate = covariate_inp;

    multiv_gp.update_train_data(predictor, outcome - des_covar_mat * covariate);
    multiv_gp.set_train_y_cov_I();
    d_outcome = multiv_gp.dim_out;
    d_predictor = multiv_gp.dim_inp;
    n_time = multiv_gp.n_training;
}

void imc_iw::update_hyperparameters(const double &alpha, const double &rho)
{
    multiv_gp.update_hyperparameters(alpha, rho);
}

void imc_iw::calc_posterior_parameters()
{
    // post_cov_scale = cov_scale + diff * inv_k * diff.t();
    // arma::mat diff = outcome - multiv_gp.train_mu -
    //   covar_mean_prior * covariate;
    arma::mat gp_train_cov = multiv_gp.get_marginal_train_cov_chol() *
                             multiv_gp.get_marginal_train_cov_chol().t();
    sigma_y = gp_train_cov +
              covariate.t() * covar_col_cov_prior * covariate;
    sigma_y_chol = chol(sigma_y, "lower");

    covar_col_cov_post = arma::inv_sympd(
        covar_col_cov_prior_inv +
        covariate * arma::inv_sympd(gp_train_cov) * covariate.t());

    covar_col_cov_post_chol = chol(covar_col_cov_post, "lower");

    covar_mean_post = (covariate * arma::inv_sympd(gp_train_cov) *
                           (outcome - multiv_gp.train_mu).t() +
                       covar_col_cov_prior_inv * covar_mean_prior.t())
                          .t() *
                      covar_col_cov_post;

    arma::mat diff = outcome - multiv_gp.train_mu -
                     covar_mean_prior * covariate;
    arma::mat L_inv_d_T = arma::solve(arma::trimatl(sigma_y_chol),
                                      diff.t(),
                                      arma::solve_opts::fast);
    // post_cov_scale_chol = chol_rank_n_update(chol(cov_scale, "lower"), 1,
    //                                          L_inv_d_T.t());

    // post_cov_scale = post_cov_scale_chol * post_cov_scale_chol.t();
    post_cov_scale = cov_scale + L_inv_d_T.t() * L_inv_d_T;
    post_cov_scale_chol = chol(post_cov_scale, "lower");
    post_cov_df = cov_df + n_time;
}

void imc_iw::sample_joint_posterior()
{
    cov = arma::iwishrnd(post_cov_scale,
                         post_cov_df);
    multiv_gp.update_sigma(cov);

    des_covar_mat = covar_mean_post +
                    chol(cov, "lower") * des_covar_mat.randn() *
                        covar_col_cov_post_chol.t();
};

void imc_iw::sample_prior()
{
    cov = arma::iwishrnd(cov_scale, cov_df);
    multiv_gp.update_sigma(cov);

    des_covar_mat = covar_mean_prior +
                    chol(cov, "lower") * des_covar_mat.randn() *
                        covar_col_cov_prior_chol.t();
};

arma::mat imc_iw::make_gp_predictions()
{
    multiv_gp.update_test_data(predictor);
    multiv_gp.set_test_y_cov_I();
    return multiv_gp.make_test_predictions();
}

double imc_iw::log_marginal_likelihood()
{
    // double log_det_y = log_det_chol(sigma_y_chol);
    // double log_det_cov_scale = log_det_chol(cov_scale_chol);
    // double log_det_post_cov_scale = log_det_chol(post_cov_scale_chol);

    // double log_marg_likelihood =
    //     -std::log(M_PI) * (d_outcome * n_time / 2.0) -
    //     (d_outcome / 2.0) * log_det_y +
    //     (cov_df / 2.0) * log_det_cov_scale -
    //     (post_cov_df / 2.0) * log_det_post_cov_scale +
    //     log_mvgamma((post_cov_df / 2.0), d_outcome) -
    //     log_mvgamma((cov_df / 2.0), d_outcome);

    return log_dmatrixt(
        sigma_y_chol, cov_scale_chol, post_cov_scale_chol, cov_df, post_cov_df);
}
