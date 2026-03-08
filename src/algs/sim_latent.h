#ifndef SIM_LATENT_H
#define SIM_LATENT_H

// [[Rcpp::depends(RcppArmadillo)]]
#include "matniw_gp_class.h"
#include "gp_model_base_class.h"
#include "hsgp_class.h"
#include "imc_gp_class.h"
#include "imc_iw_class.h"
#include "base_structs.h"
#include "derived_structs.h"

arma::mat sim_latent(
    const arma::mat &covariate, // Covariate data
    const uint &n_time,         // Number of time-points
    const uint &d_lat,          // State dimensions
    hsgp_approx &hsgp,
    const arma::vec &t0_mean,   // Latent mean at t0
    const arma::mat &t0_cov,    // Latent mean at t0
    const arma::mat &trans_mat, // State transition matrix
    const arma::mat &lat_covar, // State transition matrix
    const arma::mat &dyn_cov    // Dynamic error matrix
);

arma::mat sim_latent(
    const arma::mat &covariate,
    const uint &n_time,
    const uint &d_lat,
    const imc_gp &gp,
    const arma::vec &t0_mean, // Latent mean at t0
    const arma::mat &t0_cov,  // Latent mean at t0
    const arma::mat &lat_covar,
    const arma::mat &dyn_cov
);

#endif