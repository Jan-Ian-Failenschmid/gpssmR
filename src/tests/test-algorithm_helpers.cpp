// [[Rcpp::depends(RcppArmadillo)]]

#include <testthat.h>
#include <RcppArmadillo.h>
#include "test_helper.h"
#include "linear_algebra.h"
#include "resampling.h"
#include "pgas.h"

context("C++ Algorithm helpers")
{
    test_that("Resampling helpers produce expected results")
    {
        const double tol = 1e-10;

        arma::rowvec log_weights = {-1.0, 0.0, 1.0};
        softmax(log_weights);
        arma::rowvec expected_softmax = arma::exp(arma::rowvec({-2.0, -1.0, 0.0}));
        expected_softmax /= arma::sum(expected_softmax);
        expect_true(compare_mat(log_weights, expected_softmax, tol));

        arma::urowvec indices = systematic_resampling(
            arma::rowvec({0.0, 0.0, 1.0}), 4);
        expect_true(arma::all(indices == 2));

        std::vector<int> data = {10, 20, 30, 40};
        std::vector<int> expected = {30, 10, 40};
        std::vector<int> resampled = resample_std_vector(
            data, arma::uvec({2, 0, 3}));
        expect_true(resampled == expected);
    }
}