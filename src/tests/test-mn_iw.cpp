// [[Rcpp::depends(RcppArmadillo)]]

#include <testthat.h>
#include <RcppArmadillo.h>
#include "t_helper.h"
#include "linear_algebra.h"
#include "matniw_model_class.h"
#include "base_structs.h"
#include "derived_structs.h"

context("C++ Matrix-normal-inverse-Wishart")
{
    test_that("Matrix-normal-inverse-Wishart")
    {
        // Data generation
        set_r_seed(1);
        arma::uword n = 50;
        arma::uword d = 2;
        arma::mat X(d, n, arma::fill::randn);
        arma::mat covariate(0, n, arma::fill::zeros);

        arma::mat tans_mat(d, d, arma::fill::randn);
        arma::mat covar_mat(d, 0, arma::fill::zeros);

        arma::mat tans_mat_const(d, d, arma::fill::zeros);
        tans_mat_const.fill(arma::datum::nan);
        arma::mat covar_mat_const(d, 0, arma::fill::zeros);
        covar_mat_const.fill(arma::datum::nan);

        arma::mat tans_mat_mean(d, d, arma::fill::randn);
        arma::mat covar_mat_mean(d, 0, arma::fill::zeros);

        arma::mat col_cov = identity(d);
        arma::mat covar_col_cov = identity(0);

        arma::mat cov = identity(d);

        arma::uword cov_df = 4;
        arma::mat cov_scale = identity(d);

        arma::mat Y(d, n, arma::fill::randn);
        Y = chol(cov, "lower") * Y;
        Y += tans_mat * X;

        Dataset data;
        data.outcome = &Y;
        data.predictors = {&X, &covariate};

        // Computation
        matniw_model mn_iw_model1(tans_mat_const, covar_mat_const, tans_mat_mean, covar_mat_mean, col_cov, covar_col_cov, cov_df,
                                  cov_scale);

        mn_iw_model1.update_data(Y, X, covariate);
        mn_iw_model1.calc_posterior_parameters();

        set_r_seed(2);
        mn_iw_model1.sample_joint_posterior();

        auto mn = std::make_unique<mn_regression_model_>(
            tans_mat_mean, col_cov, identity(d), 0);

        auto iw = std::make_unique<iw_model_>(cov_df, cov_scale);

        mn_iw_model_ mn_iw_modeld(std::move(mn), std::move(iw));

        mn_iw_modeld.set_data(data, arma::mat(d, n, arma::fill::zeros),
                              identity(n));
        mn_iw_modeld.calc_posterior_parameters();
        set_r_seed(2);
        mn_iw_modeld.sample_posterior();

        // Expectation
        // Posterior means
        Rcpp::Rcerr.precision(15);
        // mn_iw_modeld.mn->param_posterior.raw_print(Rcpp::Rcerr);
        // mn_iw_model1.post_des_mat_mean.raw_print(Rcpp::Rcerr);

        arma::mat posterior_mean = {
            {0.314975546604197, -0.0291691334747645},
            {-0.0702684735339689, -1.59732729056066}};
        expect_true(
            compare_mat(mn_iw_modeld.mn->param_posterior,
                        posterior_mean, 1e-10));

        // Posterior column covariance
        // mn_iw_modeld.mn->col_cov_posterior.raw_print(Rcpp::Rcerr);
        // mn_iw_model1.post_col_cov.raw_print(Rcpp::Rcerr);
        arma::mat posterior_col_cov = {
            {0.0225818201400751, -0.000285562480808105},
            {-0.000285562480808105, 0.0173367124868897}};

        expect_true(
            compare_mat(mn_iw_modeld.mn->col_cov_posterior,
                        posterior_col_cov, 1e-10));

        // Posterior covariance scale
        // mn_iw_modeld.iw->cov_scale_posterior.raw_print(Rcpp::Rcerr);
        // mn_iw_model1.post_cov_scale.raw_print(Rcpp::Rcerr);
        arma::mat posterior_cov_scale = {
            {61.2714698014426, 4.82661921617577},
            {4.82661921617577, 31.6280512784027}};

        expect_true(
            compare_mat(mn_iw_modeld.iw->cov_scale_posterior,
                        posterior_cov_scale, 1e-10));

        // Posterior parameter sample
        // mn_iw_modeld.mn->get_param().raw_print(Rcpp::Rcerr);
        // mn_iw_model1.des_mat.raw_print(Rcpp::Rcerr);
        arma::mat posterior_param_sample = {
            {0.52023808822878, 0.24116544073709},
            {-0.119311456861801, -1.37475988508342}};

        expect_true(
            compare_mat(mn_iw_modeld.mn->get_param(),
                        posterior_param_sample, 1e-10));

        // Posterior covariance sample
        // mn_iw_modeld.iw->get_cov().raw_print(Rcpp::Rcerr);
        // mn_iw_model1.cov.raw_print(Rcpp::Rcerr);
        arma::mat posterior_cov_sample = {
            {1.17525222674298, -0.177650138700326},
            {-0.177650138700326, 1.01103141490566}};

        expect_true(
            compare_mat(mn_iw_modeld.iw->get_cov(),
                        posterior_cov_sample, 1e-10));

        // Marginal log-likelihood
        expect_true(
            compare_double(mn_iw_modeld.log_marginal_likelihood(),
                           -148.11805280905,
                           1e-10));
    }
}