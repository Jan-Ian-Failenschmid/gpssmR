#ifndef MATNIW_MODEL_CLASS_H
#define MATNIW_MODEL_CLASS_H

// [[Rcpp::depends(RcppArmadillo)]]
#include <RcppArmadillo.h>
#include "niw_base_class.h"

struct matniw_model : public normal_inverse_wishart_base
{
    // Model specific prior
    arma::mat prior_des_mat_mean;
    arma::mat prior_precision;
    arma::mat prior_col_cov;
    arma::mat prior_precision_chol;
    arma::mat prior_col_cov_chol;

    // Model specific posterior
    arma::mat post_col_cov;
    arma::mat post_col_cov_chol;
    arma::mat post_precision_chol;
    arma::mat post_des_mat_mean;

    // Conditioning
    arma::vec cond_mu;
    arma::mat cond_sigma;
    arma::mat chol_cond_sigma;
    arma::vec cond_sample;

    // Sufficient statistics
    arma::mat phi;
    arma::mat psi;
    arma::mat sigma;

    bool need_cond;

    // Constructor
    matniw_model(
        const arma::mat &des_int_mat_const,
        const arma::mat &des_covar_mat_const,
        const arma::mat &des_int_mat_mean,
        const arma::mat &des_covar_mat_mean,
        const arma::mat &des_int_mat_col_cov,
        const arma::mat &des_covar_mat_col_cov,
        uint cov_df,
        const arma::mat &cov_scale);

    void cond_mv_normal(const arma::vec &mu, const arma::mat &sigma);
    void sample_cond_mv_normal();
    void sample_matrix_normal(const arma::mat &des_mat_mean,
                              const arma::mat &col_cov_chol);
    void update_mean(
        const arma::mat &des_int_mat_mean, // Means
        const arma::mat &des_covar_mat_mean);
    void update_col_cov(
        const arma::mat &des_int_mat_col_cov,
        const arma::mat &des_covar_mat_col_cov);
    void sample_prior();
    void sample_prior_without_cov();
    void calc_posterior_parameters();
    void sample_joint_posterior();
    void sample_posterior_des_mat_cond_cov();
    double log_marginal_likelihood();
};

#endif