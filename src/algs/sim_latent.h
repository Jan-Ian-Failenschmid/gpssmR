#ifndef SIM_LATENT_H
#define SIM_LATENT_H

#include <RcppArmadillo.h>
#include "hsgp_struct.h"
#include "imc_gp_struct.h"

arma::mat sim_latent(
    const arma::mat &covariate, // Covariate data
    const arma::uword &n_time,         // Number of time-points
    const arma::uword &d_lat,          // State dimensions
    const hsgp_approx &hsgp,
    const arma::vec &t0_mean,   // Latent mean at t0
    const arma::mat &t0_cov,    // Latent mean at t0
    const arma::mat &trans_mat, // State transition matrix
    const arma::mat &lat_covar, // State transition matrix
    const arma::mat &dyn_cov    // Dynamic error matrix
);

arma::mat sim_latent(
    const arma::mat &covariate,
    const arma::uword &n_time,
    const arma::uword &d_lat,
    const imc_gp &gp,
    const arma::vec &t0_mean, // Latent mean at t0
    const arma::mat &t0_cov,  // Latent mean at t0
    const arma::mat &lat_covar,
    const arma::mat &dyn_cov
);

#endif