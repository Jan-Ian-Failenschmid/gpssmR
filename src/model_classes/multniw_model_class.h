#ifndef MULTNIW_MODEL_CLASS_H
#define MULTNIW_MODEL_CLASS_H

// [[Rcpp::depends(RcppArmadillo)]]
#include <RcppArmadillo.h>
#include "niw_base_class.h"

struct multniw_model : public normal_inverse_wishart_base
{
    // Model specific prior
    arma::vec prior_mean;
    arma::mat prior_cov;
    arma::mat prior_inv_cov;

    // Model specific posterior
    arma::mat post_cov;
    arma::vec post_mean;

    // Helper variables
    arma::uvec reorder_idx;
    arma::uvec free_idx;
    arma::uvec fix_idx;
    arma::uword n_free;
    arma::vec fix_par_vec;

    arma::mat D;
    arma::vec d;

    arma::mat F_raw;
    arma::mat F_reorder;
    arma::mat F_fixed;
    arma::mat F_free;
    arma::vec z;
    arma::mat identity;
    // Sufficient statistics

    // Constructor
    multniw_model(
        const arma::mat &des_int_mat_const,
        const arma::mat &des_covar_mat_const,
        const arma::vec &prior_int_mean,
        const arma::vec &prior_covar_mean,
        const arma::mat &prior_int_cov,
        const arma::mat &prior_covar_cov,
        uint cov_df,
        const arma::mat &cov_scale);

    void update_prior_mean(
        const arma::vec &prior_int_mean,
        const arma::vec &prior_covar_mean);
    void update_prior_cov(const arma::mat &prior_int_cov,
                          const arma::mat &prior_covar_cov);
    void sample_prior();
    void calc_posterior_parameters();
    void sample_joint_posterior();
    void sample_posterior_des_mat_cond_cov();
};

#endif