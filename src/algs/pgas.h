#ifndef PGAS_H
#define PGAS_H

// [[Rcpp::depends(RcppArmadillo)]]
#include "pgas.h"
#include "imc_gp_struct.h"
#include "hsgp_struct.h"
#include "base_structs.h"
#include "derived_structs.h"
#include "resampling.h"

// PGAS Markov kernel
arma::mat pgas(
    const arma::mat &y,
    const arma::mat &covariate_dyn,
    const arma::mat &covariate_meas,
    const int &n_particles,
    const int &n_time,
    const int &d_lat,
    hsgp_approx &hsgp,
    const arma::mat &x_ref,
    const arma::vec &t0_mean,
    const arma::mat &t0_cov,
    const arma::mat &trans_mat,
    const arma::mat &lat_covar,
    const arma::mat &dyn_cov,
    const arma::mat &des_mat,
    const arma::mat &meas_covar,
    const arma::mat &meas_cov
);

arma::mat pgas(
    const arma::mat &y,
    const arma::mat &covariate_dyn,
    const arma::mat &covariate_meas,
    const int &n_particles,
    const int &n_time,
    const int &d_lat,
    imc_gp &gp,
    const arma::mat &x_ref,
    const arma::vec &t0_mean,
    const arma::mat &t0_cov,
    const arma::mat &lat_covar,
    const arma::mat &dyn_cov,
    const arma::mat &des_mat,
    const arma::mat &meas_covar,
    const arma::mat &meas_cov
);

// Measurement model
arma::mat meas_model(
    const arma::mat &x,         // Matrix of states D * T or D * N
    const arma::vec &covariate, // Matrix of covariates D * T or D * N
    const arma::mat &des_mat,   // Design Matrix
    const arma::mat &meas_covar // Covariate effect
);

// Transition moodel
arma::mat transit_model(
    const arma::mat &x,         // Vector of states D * 1 or
    const arma::vec &covariate, // Vector of covariates D * 1
    const arma::mat &trans_mat, // Transition Matrix
    const arma::mat &lat_covar  // Covariate
);

#endif