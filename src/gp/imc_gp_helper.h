#ifndef IMC_GP_HELPER_H
#define IMC_GP_HELPER_H

// [[Rcpp::depends(RcppArmadillo)]]

// Squared Exponential Kernel
arma::mat gp_covariance_multi(const arma::mat &x1, const arma::mat &x2,
                              double length_scale,
                              double signal_variance);
arma::mat gp_covariance_multi(const arma::mat &x,
                              double length_scale,
                              double signal_variance);

#endif