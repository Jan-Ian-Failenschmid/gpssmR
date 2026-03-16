#ifndef LINEAR_ALGEBRA_H
#define LINEAR_ALGEBRA_H

#include <RcppArmadillo.h>

// Matrix vector transformation
arma::vec mat2vec(const arma::mat &matrix, const arma::mat &mask);
arma::mat vec2mat(const arma::vec &vector, const arma::mat &mask);
arma::uvec reorder_row2col(arma::uword n_rows, arma::uword n_cols);

// diag join
arma::mat diag_join(const arma::mat &A, const arma::mat &B);

// Make symmetric
void make_symmetric(arma::mat &X);

// Expand grid
arma::mat expand_grid_2d(const arma::vec &a, const arma::vec &b);

// Jitter matrix
void jitter_mat(arma::mat &K, double delta);

// Stabailized matrix inverse
void fast_inv(arma::mat &cov_inv, const arma::mat &cov_);
void stabalized_inv(arma::mat &cov, arma::mat &cov_inv, const arma::mat &cov_);

// Construct covariance matrix from cholesky pointer
void construct_cov(arma::mat &cov, const arma::mat &cov_chol_ptr);

// Identity matrix
arma::mat identity(arma::uword n);

// SYMPD Matrix log determinant
double log_det_sympd_cust(const arma::mat &X);
double log_det_chol(const arma::mat &L);

// Triangular solve
arma::mat chol_left_solve(const arma::mat &chol, const arma::mat &v);

// Cholesky updates
arma::mat chol_of_sum(const arma::mat &L1, const arma::mat &L2);

void chol_rank_one_update(
    arma::mat &L,
    double c,
    arma::vec x);
arma::mat chol_rank_n_update(
    arma::mat L,
    double c,
    arma::mat X);
void chol_rank_n_update_ip(
    arma::mat &L,
    double c,
    const arma::mat &X);
void add_cholesky_lower(
    arma::mat &L,
    const arma::mat &lower_block,
    const arma::mat &diag_block);

#endif