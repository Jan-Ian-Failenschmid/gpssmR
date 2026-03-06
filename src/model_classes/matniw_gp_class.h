#ifndef MATNIW_GP_CLASS_H
#define MATNIW_GP_CLASS_H

// [[Rcpp::depends(RcppArmadillo)]]
#include <RcppArmadillo.h>
#include "matniw_model_class.h"
#include "gp_model_base_class.h"
#include "hsgp_class.h"

struct matniw_gp_model : public matniw_model, public gp_model_base
{
    hsgp_approx hsgp;
    arma::mat des_covar_mat_col_cov;

    matniw_gp_model(
        const arma::mat &basis_indices,
        const arma::vec &boundary_factor,
        const arma::vec &hyperparameters,
        const arma::mat &des_int_mat_const,
        const arma::mat &des_covar_mat_const,
        const arma::mat &des_int_mat_mean,
        const arma::mat &des_covar_mat_mean,
        const arma::mat &des_covar_mat_col_cov_inp,
        uint cov_df,
        const arma::mat &cov_scale);
    void update_data(const arma::mat &outcome_inp,
                     const arma::mat &internal_inp,
                     const arma::mat &covariate_inp);
    void update_hyperparameters(const double &alpha, const double &rho);
    void calc_posterior_parameters() override;
    void sample_prior() override;
    void sample_joint_posterior() override;
    double log_marginal_likelihood() override;
    arma::mat make_gp_predictions() override;

    // Declare getters
    const arma::mat &get_des_int_mat() const override
    {
        return matniw_model::des_int_mat;
    }
    const arma::mat &get_des_covar_mat() const override
    {
        return matniw_model::des_covar_mat;
    }
    const arma::mat &get_cov() const override
    {
        return matniw_model::cov;
    }
};

#endif