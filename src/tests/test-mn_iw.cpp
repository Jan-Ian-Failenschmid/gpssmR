// [[Rcpp::depends(RcppArmadillo)]]

#include <testthat.h>
#include <RcppArmadillo.h>
#include "test_helper.h"
#include "linear_algebra.h"
#include "main_helper.h"

context("C++ Matrix-normal-inverse-Wishart")
{
    test_that("Matrix-normal-inverse-Wishart matches reference values")
    {
        const double tol = 1e-10;

        set_r_seed(1);
        const arma::uword n = 50;
        const arma::uword d = 2;
        arma::mat X(d, n, arma::fill::randn);
        arma::mat covariate(0, n, arma::fill::zeros);

        arma::mat tans_mat(d, d, arma::fill::randn);
        arma::mat tans_mat_mean(d, d, arma::fill::randn);
        arma::mat covar_mat_mean(d, 0, arma::fill::zeros);

        const arma::mat col_cov = identity(d);
        const arma::mat covar_col_cov = identity(0);
        const arma::mat cov = identity(d);
        const arma::uword cov_df = 4;
        const arma::mat cov_scale = identity(d);

        arma::mat Y(d, n, arma::fill::randn);
        Y = chol(cov, "lower") * Y;
        Y += tans_mat * X;

        arma::mat col_cov_chol = chol(col_cov, "lower");
        arma::mat covar_col_cov_chol = chol(covar_col_cov, "lower");
        arma::mat cov_scale_chol = chol(cov_scale, "lower");
        arma::mat data_mean(d, n, arma::fill::zeros);
        arma::mat data_cov = identity(n);

        mn_covar_wrapper model_wrapper(
            &X, &covariate,
            &tans_mat_mean, &covar_mat_mean,
            &col_cov_chol, &covar_col_cov_chol);

        mn_iw_model_ model = init_mn_iw_model(
            Y,
            data_mean,
            data_cov,
            model_wrapper,
            cov_scale_chol,
            cov_df);

        model.calc_posterior_parameters();
        set_r_seed(2);
        model.sample_posterior();

        const arma::mat posterior_mean = {
            {0.314975546604197, -0.0291691334747645},
            {-0.0702684735339689, -1.59732729056066}};
        expect_true(compare_mat(model.mn->coefficient_posterior,
                                posterior_mean, tol));

        const arma::mat posterior_col_cov = {
            {0.0225818201400751, -0.000285562480808105},
            {-0.000285562480808105, 0.0173367124868897}};

        expect_true(
            compare_mat(model.mn->col_cov_posterior,
                        posterior_col_cov, tol));

        const arma::mat posterior_cov_scale = {
            {61.2714698014426, 4.82661921617577},
            {4.82661921617577, 31.6280512784027}};

        expect_true(
            compare_mat(model.iw->cov_scale_posterior,
                        posterior_cov_scale, tol));

        const arma::mat posterior_param_sample = {
            {0.52023808822878, 0.24116544073709},
            {-0.119311456861801, -1.37475988508342}};

        expect_true(
            compare_mat(model_wrapper.get_pred_param(),
                        posterior_param_sample, tol));

        const arma::mat posterior_cov_sample = {
            {1.17525222674298, -0.177650138700326},
            {-0.177650138700326, 1.01103141490566}};

        expect_true(
            compare_mat(model.iw->get_cov(),
                        posterior_cov_sample, tol));

        expect_true(
            compare_double(model.log_marginal_likelihood(),
                           -148.11805280905,
                           tol));
    }
}
