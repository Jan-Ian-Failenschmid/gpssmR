// [[Rcpp::depends(RcppArmadillo)]]

#include <testthat.h>
#include <RcppArmadillo.h>
#include "test_helper.h"
#include "linear_algebra.h"
#include "derived_structs.h"

context("C++ Model wrappers")
{
    test_that("Matrix-normal covariate wrapper combines data and priors correctly")
    {
        const double tol = 1e-10;

        arma::mat predictor = {
            {1.0, 2.0, 3.0},
            {4.0, 5.0, 6.0}};
        arma::mat covariate(1, 3);
        covariate.row(0) = arma::rowvec({7.0, 8.0, 9.0});
        arma::mat pred_prior_mean = {
            {0.1, 0.2},
            {0.3, 0.4}};
        arma::mat covar_prior_mean(2, 1);
        covar_prior_mean.col(0) = arma::vec({0.5, 0.6});
        arma::mat pred_prior_cov_chol = {
            {1.0, 0.0},
            {0.2, 1.1}};
        arma::mat covar_prior_cov_chol(1, 1, arma::fill::ones);
        covar_prior_cov_chol *= 0.7;

        mn_covar_wrapper wrapper(&predictor, &covariate,
                                 &pred_prior_mean, &covar_prior_mean,
                                 &pred_prior_cov_chol, &covar_prior_cov_chol);

        arma::mat expected_data = arma::join_cols(predictor, covariate);
        arma::mat expected_mean = arma::join_rows(pred_prior_mean, covar_prior_mean);
        arma::mat expected_cov_chol =
            diag_join(pred_prior_cov_chol, covar_prior_cov_chol);

        expect_true(compare_mat(*wrapper.get_data_ptr(), expected_data, tol));
        expect_true(compare_mat(*wrapper.get_mean_ptr(), expected_mean, tol));
        expect_true(compare_mat(*wrapper.get_prior_cov_chol_ptr(),
                                expected_cov_chol, tol));

        arma::mat param = {
            {1.0, 2.0, 3.0},
            {4.0, 5.0, 6.0}};
        wrapper.set_param_ptr(&param);
        expect_true(compare_mat(wrapper.get_pred_param(),
                                param.cols(0, 1), tol));
        expect_true(compare_mat(wrapper.get_covar_param(),
                                param.cols(2, 2), tol));
    }

    test_that("Multivariate-normal covariate wrapper combines constraints and means")
    {
        const double tol = 1e-10;
        const double nan = arma::datum::nan;

        arma::mat predictor = {
            {1.0, 2.0},
            {3.0, 4.0}};
        arma::mat covariate(1, 2);
        covariate.row(0) = arma::rowvec({5.0, 6.0});
        arma::mat predictor_constraints = {
            {nan, 1.0},
            {2.0, nan}};
        arma::mat covar_constraints(2, 1);
        covar_constraints.col(0) = arma::vec({nan, 0.0});
        arma::vec predictor_prior_mean = {0.1, 0.2};
        arma::vec covar_prior_mean = {0.3};
        arma::mat predictor_prior_cov_chol = arma::diagmat(arma::vec({1.0, 2.0}));
        arma::mat covar_prior_cov_chol = arma::diagmat(arma::vec({3.0}));

        mvn_covar_wrapper wrapper(&predictor, &covariate,
                                  &predictor_constraints, &covar_constraints,
                                  &predictor_prior_mean, &covar_prior_mean,
                                  &predictor_prior_cov_chol,
                                  &covar_prior_cov_chol);

        expect_true(compare_mat(*wrapper.get_data_ptr(),
                                arma::join_cols(predictor, covariate),
                                tol));
        expect_true(compare_mat(*wrapper.get_mean_ptr(),
                                arma::join_cols(predictor_prior_mean,
                                                covar_prior_mean),
                                tol));
        expect_true(compare_mat(*wrapper.get_prior_cov_chol_ptr(),
                                diag_join(predictor_prior_cov_chol,
                                          covar_prior_cov_chol),
                                tol));
        arma::mat expected_constraints =
            arma::join_rows(predictor_constraints, covar_constraints);
        expect_true(arma::all(
            arma::find_nonfinite(*wrapper.get_constraints_ptr()) ==
            arma::find_nonfinite(expected_constraints)));
        arma::mat wrapper_constraints = *wrapper.get_constraints_ptr();
        wrapper_constraints.elem(arma::find_nonfinite(wrapper_constraints)).zeros();
        expected_constraints.elem(arma::find_nonfinite(expected_constraints)).zeros();
        expect_true(compare_mat(wrapper_constraints, expected_constraints, tol));

        arma::mat param = {
            {1.0, 2.0, 3.0},
            {4.0, 5.0, 6.0}};
        wrapper.set_param_ptr(&param);
        expect_true(compare_mat(wrapper.get_pred_param(),
                                param.cols(0, 1), tol));
        expect_true(compare_mat(wrapper.get_covar_param(),
                                param.cols(2, 2), tol));
    }
}
