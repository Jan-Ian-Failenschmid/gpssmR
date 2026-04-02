#ifndef PGAS_H
#define PGAS_H

// [[Rcpp::depends(RcppArmadillo)]]
#include <RcppArmadillo.h>

struct imc_gp;
struct hsgp_approx;

// PGAS Markov kernel
arma::mat pgas(
    const arma::mat &y,
    const arma::mat &covariate_dyn,
    const arma::mat &covariate_meas,
    const arma::uword &n_particles,
    const arma::uword &n_time,
    const arma::uword &d_lat,
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
    const arma::uword &n_particles,
    const arma::uword &n_time,
    const arma::uword &d_lat,
    imc_gp &gp,
    const arma::mat &x_ref,
    const arma::vec &t0_mean,
    const arma::mat &t0_cov,
    const arma::mat &lat_covar,
    const arma::mat &dyn_cov,
    const arma::mat &des_mat,
    const arma::mat &meas_covar,
    const arma::mat &meas_cov);

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
