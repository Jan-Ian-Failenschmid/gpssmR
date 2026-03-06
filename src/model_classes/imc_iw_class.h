#ifndef IMC_IW_H
#define IMC_IW_H

// [[Rcpp::depends(RcppArmadillo)]]
#include "imc_gp_class.h"
#include "gp_model_base_class.h"

struct imc_iw : public gp_model_base
{
    uint d_outcome;
    uint d_predictor;
    uint n_time;

    imc_gp multiv_gp;

    arma::mat outcome;
    arma::mat predictor;
    arma::mat covariate;
    arma::mat des_covar_mat;
    arma::mat des_int_mat; // Necessary for polymorphism

    arma::mat covar_mean_prior;
    arma::mat covar_mean_post;
    arma::mat covar_col_cov_prior;
    arma::mat covar_col_cov_prior_inv;
    arma::mat covar_col_cov_prior_chol;
    arma::mat covar_col_cov_post;
    arma::mat covar_col_cov_post_chol;

    arma::mat cov;

    uint cov_df;
    arma::mat cov_scale;
    arma::mat cov_scale_chol;

    uint post_cov_df;
    arma::mat post_cov_scale;
    arma::mat post_cov_scale_chol;

    arma::mat sigma_y;
    arma::mat sigma_y_chol;
    arma::mat K;

    imc_iw(
        const arma::vec &hyperparameters_inp,
        const arma::mat &covar_mean_prior_inp,
        const arma::mat &covar_col_cov_prior_inp,
        const int &cov_df_inp,
        const arma::mat &cov_scale_inp);

    void update_data(
        const arma::mat &outcome_inp,
        const arma::mat &predictor_inp,
        const arma::mat &covariate_inp);
    void update_hyperparameters(const double &alpha, const double &rho);
    void calc_posterior_parameters();
    void sample_joint_posterior() override;
    void sample_prior();
    arma::mat make_gp_predictions() override;
    double log_marginal_likelihood() override;

    const arma::mat &get_des_int_mat() const override
    {
        return des_int_mat;
    }
    const arma::mat &get_des_covar_mat() const override { return des_covar_mat; }
    const arma::mat &get_cov() const override { return cov; }
};

#endif