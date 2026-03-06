// [[Rcpp::depends(RcppArmadillo)]]
#include <RcppArmadillo.h>
#include "niw_base_class.h"
#include "multniw_model_class.h"
#include "linear_algebra.h"
#include "pdfs.h"

multniw_model::multniw_model(
    const arma::mat &des_int_mat_const,
    const arma::mat &des_covar_mat_const,
    const arma::vec &prior_int_mean,
    const arma::vec &prior_covar_mean,
    const arma::mat &prior_int_cov,
    const arma::mat &prior_covar_cov,
    uint cov_df,
    const arma::mat &cov_scale)
    : normal_inverse_wishart_base(des_int_mat_const,
                                  des_covar_mat_const,
                                  cov_df,
                                  cov_scale),
      F_raw(),
      F_reorder(),
      F_fixed(),
      F_free(),
      z()
{
    update_prior_mean(prior_int_mean, prior_covar_mean);
    update_prior_cov(prior_int_cov, prior_covar_cov);

    // prior_mean = prior_mean;
    // post_cov = prior_cov;

    reorder_idx = reorder_row2col(des_mat.n_rows, des_mat.n_cols);
    free_idx = arma::find_nonfinite(des_mat_const);
    fix_idx = arma::find_finite(des_mat_const);
    n_free = free_idx.n_elem;
    fix_par_vec = des_mat_const.elem(fix_idx);

    D.zeros(n_free, n_free);
    d.zeros(n_free);

    identity = arma::eye(des_mat.n_rows, des_mat.n_rows);
}

// Prior -------

void multniw_model::update_prior_mean(
    const arma::vec &prior_int_mean,
    const arma::vec &prior_covar_mean)
{
    if (prior_covar_mean.n_elem == 0)
    {
        prior_mean = prior_int_mean;
    }
    else
    {
        prior_mean = arma::join_cols(prior_int_mean, prior_covar_mean);
    }
}

void multniw_model::update_prior_cov(const arma::mat &prior_int_cov,
                                     const arma::mat &prior_covar_cov)
{
    if (prior_covar_cov.n_elem == 0)
    {
        prior_cov = prior_int_cov;
    }
    else
    {
        prior_cov = diag_join(prior_int_cov, prior_covar_cov);
    }
    prior_inv_cov = arma::inv_sympd(prior_cov);
}

void multniw_model::sample_prior()
{
    cov = arma::iwishrnd(cov_scale, cov_df); // Sample covariance from prior
    arma::vec prior_sample = prior_mean +
                             chol(prior_cov, "lower") *
                                 arma::randn(prior_mean.n_elem);
    fill_sample_into_des_mat(prior_sample);
};

// Posterior ---
void multniw_model::calc_posterior_parameters()
{
    // Zero auxiliary variables
    D.zeros();
    d.zeros();

    arma::mat cov_inv = arma::inv_sympd(cov);
    arma::mat identity = arma::eye(des_mat.n_rows, des_mat.n_rows);

    for (size_t t = 0; t < n_time; t++)
    {
        // Construct, reorder and split up F into fixed and free contribution
        F_raw = arma::kron(identity, predictor.col(t).t());
        F_reorder = F_raw.cols(reorder_idx);
        F_fixed = F_reorder.cols(fix_idx);
        F_free = F_reorder.cols(free_idx);

        // Remove fixed contribution from outcome
        z = outcome.col(t) - F_fixed * fix_par_vec;

        D += F_free.t() * cov_inv * F_free;
        d += F_free.t() * cov_inv * z;
    }

    D += prior_inv_cov;
    d += prior_inv_cov * prior_mean;

    post_cov = arma::inv_sympd(D);
    post_mean = post_cov * d;
};

void multniw_model::sample_joint_posterior()
{
    0;
};

void multniw_model::sample_posterior_des_mat_cond_cov()
{

    arma::vec post_sample = post_mean + chol(post_cov, "lower") *
                                            arma::randn(n_free);

    fill_sample_into_des_mat(post_sample);
}
