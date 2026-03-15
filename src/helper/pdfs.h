#ifndef PROBABILITY_DISTRIBUTIONS_H
#define PROBABILITY_DISTRIBUTIONS_H

#include <RcppArmadillo.h>

using namespace Rcpp;

// Probability distributions
// Multivariate normal distribution
double logdnorm(const arma::vec &x, const arma::vec &mu,
                const arma::mat &cov_chol);
arma::vec mat_logdnorm_kernel(const arma::mat &diff, const arma::mat &chol_cov);
arma::vec mat_logdnorm(const arma::vec &x,
                       const arma::mat &mu,
                       const arma::mat &chol_cov);
arma::vec mat_logdnorm(const arma::mat &x,
                       const arma::vec &mu,
                       const arma::mat &chol_cov);
arma::vec mat_logdnorm(const arma::mat &x,
                       const arma::mat &mu,
                       const arma::mat &chol_cov);
void mat_rnorm(arma::mat &X, const arma::mat &mu, const arma::mat &cov_chol);
arma::mat mat_rnorm(const arma::mat &mu, const arma::mat &cov_chol);
void conditional_mvnormal(
    arma::vec &cond_mu,
    arma::vec &cond_sigma,
    const arma::vec &mu,
    const arma::mat &sigma,
    const arma::vec &fixed_vals,
    const arma::uvec &fixed, // Indicator for fixed element
    const arma::uvec &free   // Indicator for free elements
);

// Matrix normal density
double logdmatnorm(
    const arma::mat &X,
    const arma::mat &mu,
    const arma::mat &row_chol,
    const arma::mat &col_chol);
void rmatnorm(
    arma::mat &X,
    const arma::mat &mu,
    const arma::mat &row_chol,
    const arma::mat &col_chol);
arma::mat rmatnorm(
    const arma::mat &mu,
    const arma::mat &row_chol,
    const arma::mat &col_chol);

// Wishart and Inverse wishart
double log_mvgamma(double a, int p);

double log_dmatrixt(
    const arma::mat &sigma_y_chol,
    const arma::mat &prior_cov_scale_chol,
    const arma::mat &post_cov_scale_chol,
    uint v_prior,
    uint v_post);

#endif