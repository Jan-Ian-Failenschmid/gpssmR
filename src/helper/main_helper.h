#ifndef INITIALIZERS_H
#define INITIALIZERS_H

#include "base_structs.h"
#include "derived_structs.h"
#include "hsgp_struct.h"
#include "imc_gp_struct.h"

mn_iw_model_ init_mn_iw_model(
    arma::mat &Y,
    arma::mat &data_mean,
    arma::mat &data_cov,
    mn_covar_wrapper &model_wrapper,
    arma::mat &cov_scale_chol,
    double cov_df);

mvn_iw_model_ init_mvn_iw_model(
    arma::mat &Y,
    arma::mat &data_mean,
    mvn_covar_wrapper &model_wrapper,
    arma::mat &cov_scale_chol,
    double cov_df);

void update_model_hyperparameters(
    const arma::vec &hyperparameters,
    gp_base &gp,
    mn_iw_model_ &model,
    mn_covar_wrapper &wrapper);

void update_model_predictor(
    const arma::mat &raw_pred,
    gp_base &gp,
    mn_iw_model_ &model,
    mn_covar_wrapper &wrapper);

void run_sim_latent(
    arma::mat &x,
    const arma::mat &covariate_dyn,
    const gp_base &gp,
    const mn_iw_model_ &dyn_model,
    const mn_covar_wrapper &dyn_wrapper,
    const arma::vec &t0_mean,
    const arma::mat &t0_cov,
    uint pg_rep);

void run_pgas(
    const arma::mat &y,
    arma::mat &x,
    const arma::mat &covariate_dyn,
    const arma::mat &covariate_meas,
    uint n_particles,
    const gp_base &gp,
    const mn_iw_model_ &dyn_model,
    const mn_covar_wrapper &dyn_wrapper,
    const mvn_iw_model_ &meas_model,
    const mvn_covar_wrapper &meas_wrapper,
    const arma::vec &t0_mean,
    const arma::mat &t0_cov,
    uint pg_rep);

#endif