#ifndef PGAS_H
#define PGAS_H

// [[Rcpp::depends(RcppArmadillo)]]
#include "pgas.h"
#include "imc_gp_class.h"
#include "gp_model_base_class.h"
#include "matniw_model_class.h"
#include "niw_base_class.h"
#include "base_structs.h"
#include "derived_structs.h"

arma::mat pgas(
    const arma::mat &y,
    const arma::mat &covariate_dyn,
    const arma::mat &covariate_meas,
    const uint &n_particles,
    const uint &n_time,
    const uint &d_lat,
    gp_model_base &dyn_model,
    normal_inverse_wishart_base &meas_model,
    const arma::mat &x_prev,
    const arma::vec &t0_mean,
    const arma::mat &t0_cov);

arma::mat pgas_markov(
    const arma::mat &y,              // Observation matrix y
    const arma::mat &covariate_dyn,  // Covariate data
    const arma::mat &covariate_meas, // Covariate data
    const int &n_particles,          // Number of particles
    const int &n_time,               // Number of time-points
    const int &d_lat,                // State dimensions
    hsgp_approx &hsgp,
    const arma::mat &x_ref,      // Reference sample from last iteration
    const arma::vec &t0_mean,    // Latent mean at t0
    const arma::mat &t0_cov,     // Latent mean at t0
    const arma::mat &trans_mat,  // State transition matrix
    const arma::mat &lat_covar,  // State transition matrix
    const arma::mat &dyn_cov,    // Dynamic error matrix
    const arma::mat &des_mat,    // Design Matrix
    const arma::mat &meas_covar, // State transition matrix
    const arma::mat &meas_cov    // Measurement Error Matrix
);

arma::mat mpgas_markov(
    const arma::mat &y, // Observation matrix y
    const arma::mat &covariate_dyn,
    const arma::mat &covariate, // Covariate data
    const int &n_particles,     // Number of particles
    const int &n_time,          // Number of time-points
    const int &d_lat,           // State dimensions
    const imc_gp &gp_inp,
    const arma::mat &x_ref,      // Reference sample from last iteration
    const arma::vec &t0_mean,    // Latent mean at t0
    const arma::mat &t0_cov,     // Latent mean at t0
    const arma::mat &lat_covar,  // State transition matrix
    const arma::mat &dyn_cov,    // Dynamic error matrix
    const arma::mat &des_mat,    // Design Matrix
    const arma::mat &meas_covar, // State transition matrix
    const arma::mat &meas_cov    // Measurement Error Matrix
);

// PGAS Markov kernel
arma::mat pgas(
    const arma::mat &y,
    const arma::mat &covariate_dyn,
    const arma::mat &covariate_meas,
    const int &n_particles,
    const int &n_time,
    const int &d_lat,
    const hsgp_approx &hsgp,
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
    const imc_gp &gp,
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