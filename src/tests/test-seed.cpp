// [[Rcpp::depends(RcppArmadillo)]]

#include <testthat.h>
#include <RcppArmadillo.h>
#include "t_helper.h"

context("C++ R Seed")
{
  test_that("Set R seed to Armadillo")
  {
    set_r_seed(1);
    arma::mat A(4, 4, arma::fill::randn);

    set_r_seed(1);
    arma::mat B(4, 4, arma::fill::randn);

    expect_true(arma::approx_equal(A, B, "absdiff", 1e-10));
  }
}