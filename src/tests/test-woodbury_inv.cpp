// [[Rcpp::depends(RcppArmadillo)]]
#include <testthat.h>
#include <RcppArmadillo.h>
#include "test_helper.h"
#include "linear_algebra.h"

context("C++ Test-woodburry inverse")
{
    test_that("Test-woodburry inverse")
    {
        set_r_seed(1);

        arma::mat A = arma::iwishrnd(identity(2), 5.0);
        arma::mat B = arma::iwishrnd(identity(5), 6.0);
        arma::mat U(2, 5, arma::fill::randn);

        arma::mat A_inv = arma::inv_sympd(A);
        arma::mat B_inv = arma::inv_sympd(B);

        arma::mat res1 = arma::inv_sympd(A + U * B * U.t());
        arma::mat res2 = woodbury_inv(A_inv, U, B_inv);

        expect_true(compare_mat(res1, res2, 1e-10));
    };
};