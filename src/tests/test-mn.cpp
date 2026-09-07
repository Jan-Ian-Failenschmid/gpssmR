// [[Rcpp::depends(RcppArmadillo)]]

#include <testthat.h>
#include <RcppArmadillo.h>
#include "test_helper.h"
#include "linear_algebra.h"
#include "main_helper.h"

context("C++ Matrix-normal models")
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
        arma::mat cov_scale_chol;
        arma::mat data_mean(d, n, arma::fill::zeros);
        arma::mat data_cov = identity(n);

        mn_covar_wrapper model_wrapper(
            &X, &covariate,
            &tans_mat_mean, &covar_mat_mean,
            &col_cov_chol, &covar_col_cov_chol);

        const Rcpp::List cov_list = Rcpp::List::create(
            _["prior_scale"] = cov_scale,
            _["prior_df"] = cov_df,
            _["is_fixed"] = false
        );
        
        mn_iw_model_ model = init_mn_iw_model(
            Y,
            data_mean,
            data_cov,
            model_wrapper,
            cov_scale_chol,
            cov_list);

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
        
        expect_true(compare_mat(
            dynamic_cast<iw_base*>(model.covariance.get())->cov_scale_posterior,
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
            compare_mat(model.covariance->get_cov(),
                        posterior_cov_sample, tol));

        expect_true(
            compare_double(model.log_marginal_likelihood(),
                           -148.11805280905,
                           tol));
    }
    test_that("Matrix-normal with fixed covariance matches direct calculations")
    {
        const double tol = 1e-10;
        set_r_seed(1);

        const arma::uword n = 50;
        const arma::uword d = 2;
        arma::mat X(d, n, arma::fill::randn);
        arma::mat covariate(0, n, arma::fill::zeros);

        arma::mat trans_mat(d, d, arma::fill::randn);
        arma::mat trans_mat_mean(d, d, arma::fill::randn);
        arma::mat covar_mat_mean(d, 0, arma::fill::zeros);

        const arma::mat col_cov = identity(d);
        const arma::mat covar_col_cov = identity(0);
        arma::mat col_cov_chol = chol(col_cov, "lower");
        arma::mat covar_col_cov_chol = chol(covar_col_cov, "lower");

        // Use non-identity covariance for robustness
        const arma::mat fixed_cov = {
           {1.2, 0.25},
           {0.25, 0.8}
        };

        arma::mat Y(d, n, arma::fill::randn);
        Y = chol(fixed_cov, "lower") * Y;
        Y += trans_mat * X;

        // Not used by fixed_covariance, but needed for initializer.
        arma::mat cov_scale_chol;

        arma::mat data_mean(d, n, arma::fill::zeros);
        arma::mat data_cov = identity(n);

        mn_covar_wrapper model_wrapper(
            &X, &covariate,
            &trans_mat_mean, &covar_mat_mean,
            &col_cov_chol, &covar_col_cov_chol);

        const Rcpp::List cov_list =
            Rcpp::List::create(
                _["value"] = fixed_cov,
                _["is_fixed"] = true
            );


        mn_iw_model_ model = init_mn_iw_model(Y, 
                data_mean, data_cov,
                model_wrapper,
                cov_scale_chol, cov_list);

        model.calc_posterior_parameters();

        // Direct matrix normal calculations
        const arma::mat data_cov_inv = identity(n);
        const arma::mat col_cov_inv = arma::inv_sympd(col_cov);

        const arma::mat sigma = X * data_cov_inv * X.t();
        const arma::mat psi = (Y - data_mean) * data_cov_inv * X.t();
        const arma::mat expected_col_cov_inv = sigma + col_cov_inv;
        
        const arma::mat expected_col_cov = arma::inv_sympd( 
            expected_col_cov_inv);
        const arma::mat expected_mean = (trans_mat_mean * col_cov_inv + psi) *
            expected_col_cov;

        expect_true(compare_mat(model.mn->coefficient_posterior, expected_mean,
                tol));

        expect_true(compare_mat(model.mn->col_cov_posterior, expected_col_cov,
                tol));

        // Fixed covariance does not change
        expect_true(compare_mat(model.get_cov(), fixed_cov, tol));

        model.covariance->sample_prior();
        expect_true(compare_mat(model.get_cov(), fixed_cov, tol));

        model.covariance->sample_posterior();
        expect_true(compare_mat(model.get_cov(), fixed_cov, tol));

        // Posterior matrix normal sample
        const arma::mat expected_col_cov_chol = arma::chol(
            expected_col_cov, "lower");
        const arma::mat fixed_cov_chol = arma::chol(
            fixed_cov, "lower");

        arma::mat expected_param_sample(d, d);

        set_r_seed(2);
        rmatnorm(expected_param_sample, expected_mean, 
            fixed_cov_chol, expected_col_cov_chol);

        set_r_seed(2);
        model.sample_posterior();

        expect_true(compare_mat(
                model_wrapper.get_pred_param(), expected_param_sample, tol));

        // Marginal log-likelihood 
        // Y ~ MN(data_mean + M0 X, Sigma, data_cov + X' V0 X)
        const arma::mat marginal_mean = data_mean + trans_mat_mean * X;
        const arma::mat marginal_col_cov = data_cov + X.t() * col_cov * X;

        const arma::mat marginal_col_cov_chol = arma::chol(
            marginal_col_cov, "lower");

        const double expected_log_marginal_likelihood =
            logdmatnorm(Y, marginal_mean, fixed_cov_chol,
                marginal_col_cov_chol);

        expect_true(compare_double(
                model.log_marginal_likelihood(),
                expected_log_marginal_likelihood, tol));
                
        // Covariance still invariant after sampling complete model
        expect_true(compare_mat(model.get_cov(), fixed_cov, tol));
    }
}
