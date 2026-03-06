// [[Rcpp::depends(RcppArmadillo)]]
#include <RcppArmadillo.h>
#include "matniw_model_class.h"
#include "gp_model_base_class.h"
#include "hsgp_class.h"
#include "matniw_gp_class.h"

matniw_gp_model::matniw_gp_model(
    const arma::mat &basis_indices,
    const arma::vec &boundary_factor,
    const arma::vec &hyperparameters,
    const arma::mat &des_int_mat_const,
    const arma::mat &des_covar_mat_const,
    const arma::mat &des_int_mat_mean,
    const arma::mat &des_covar_mat_mean,
    const arma::mat &des_covar_mat_col_cov_inp,
    uint cov_df,
    const arma::mat &cov_scale)
    : matniw_model(
          des_int_mat_const,
          des_covar_mat_const,
          des_int_mat_mean,
          des_covar_mat_mean,
          arma::eye(1, 1),
          des_covar_mat_col_cov_inp,
          cov_df,
          cov_scale),
      hsgp(basis_indices, boundary_factor),
      des_covar_mat_col_cov(des_covar_mat_col_cov_inp)
{
    hsgp.update_hyperparameters(hyperparameters[0],
                                hyperparameters[1]);

    update_col_cov(hsgp.scale(), des_covar_mat_col_cov);
}

void matniw_gp_model::update_data(const arma::mat &outcome_inp,
                                  const arma::mat &internal_inp,
                                  const arma::mat &covariate_inp)
{
    predictor = join_cols(hsgp.basis_functions(internal_inp), covariate_inp);
    outcome = outcome_inp;
    n_time = outcome.n_cols;
};

void matniw_gp_model::update_hyperparameters(const double &alpha, 
    const double &rho)
{
    hsgp.update_hyperparameters(alpha, rho);

    update_col_cov(hsgp.scale(), des_covar_mat_col_cov);
}

void matniw_gp_model::calc_posterior_parameters() 
{
    matniw_model::calc_posterior_parameters();
};

void matniw_gp_model::sample_prior() 
{
    matniw_model::sample_prior();
};

void matniw_gp_model::sample_joint_posterior() 
{
    matniw_model::sample_joint_posterior();
};

double matniw_gp_model::log_marginal_likelihood() 
{
    return matniw_model::log_marginal_likelihood();
};

arma::mat matniw_gp_model::make_gp_predictions()
{
    return matniw_model::des_int_mat * hsgp.phi;
}