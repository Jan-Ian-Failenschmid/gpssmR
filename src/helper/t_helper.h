#ifndef TESTS_HELPER_H
#define TESTS_HELPER_H

// Seeding
void set_r_seed(double seed);

// Compare double up to precision
bool compare_double(double x, double y, double precision);

// Compare matrix elemnt wise up to precision
bool compare_mat(const arma::mat &X, const arma::mat &Y, double precision);

#endif