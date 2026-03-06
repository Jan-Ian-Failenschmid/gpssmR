#ifndef HSGP_HELPER_H
#define HSGP_HELPER_H

// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::depends(RcppProgress)]]

// Hilbert-space approximate Gaussian processes
arma::mat gp_sqrt_lambda_nd_vec(
    const arma::rowvec &L, // Boundry factor in each dimension
    const arma::mat &m     // Basis function indicator in each dimension
);

arma::vec gp_spdf_nd_vec(
    const arma::mat &Lambda, // Matrix where each row is a lambda vector
    const double &alpha,     // Marginal variance
    const double &rho        // Length scale
);

// This currently includes some redundant calculations
arma::rowvec gp_phi_nD(
    const arma::vec &L,           // Boundry factor in each dimension
    const arma::vec &sqrt_lambda, // Sqrt lambda
    const arma::mat &X_mat        // Input matrix
);

#endif