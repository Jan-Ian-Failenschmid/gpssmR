// [[Rcpp::depends(RcppArmadillo)]]

#include <testthat.h>
#include <RcppArmadillo.h>
#include "test_helper.h"
#include "linear_algebra.h"
#include "imc_gp_struct.h"
#include "main_helper.h"

context("C++ GP Matrix-normal-inverse-Wishart")
{
    test_that("GP Matrix-normal-inverse-Wishart")
    {
        // Data generation
        set_r_seed(1);
        arma::uword n = 50;
        arma::uword d1 = 2;
        arma::uword d2 = 2;

        arma::mat X(d1, n, arma::fill::randn);
        arma::mat covariate(d2, n, arma::fill::randn);

        arma::vec hyperparameters = {5.0, 1.0};

        arma::mat covar_mat(d1, d2, arma::fill::randn);

        arma::mat covar_mat_const(d1, d2, arma::fill::zeros);
        covar_mat_const.fill(arma::datum::nan);

        arma::mat covar_mat_mean(d1, d2, arma::fill::randn);

        arma::mat covar_prior_col_cov = identity(d2);

        arma::mat cov = 0.5 * identity(d1);

        arma::uword cov_df = 4;
        arma::mat cov_scale = identity(d1);

        arma::mat Y(d1, n, arma::fill::randn);
        Y = arma::chol(cov, "lower") * Y;
        Y += arma::tanh(X) + covar_mat * covariate;
        arma::mat dyn_mat_mean = Y;
        dyn_mat_mean.zeros();
        arma::mat data_mean = Y;
        data_mean.zeros();

        arma::mat identity_pred = identity(n);

        // Dataset data;
        // data.outcome = &Y;
        // data.predictors = {&X, &covariate, &identity_pred};

        // Computation
        // New model
        arma::mat covar_col_cov_chol = chol(covar_prior_col_cov, "lower");
        arma::mat cov_scale_chol = chol(cov_scale, "lower");
        arma::mat data_cov = identity(n);

        auto gp = std::make_unique<imc_gp>();
        // gp->update_train_data(X, Y); // Y is not really needed here
        gp->set_hyperparameters(5.0, 1.0);
        gp->update_predictor(X);

        mn_covar_wrapper model_wrapper(
            gp->get_predictor_ptr(), &covariate,
            &dyn_mat_mean, &covar_mat_mean,
            gp->get_cov_chol_ptr(), &covar_col_cov_chol);

        mn_iw_model_ mn_iw_modeld = init_mn_iw_model(
            Y,
            data_mean,
            data_cov,
            model_wrapper,
            cov_scale_chol,
            cov_df);

        mn_iw_modeld.calc_posterior_parameters();

        // Expectation
        Rcpp::Rcerr.precision(15);

        // Posterior covariance scale
        arma::mat posterior_cov_scale = {
            {23.0381917807862, -1.4203057830965},
            {-1.4203057830965, 27.6275728166084}};
        // mn_iw_modeld.iw->cov_scale_posterior.raw_print(Rcpp::Rcerr);
        expect_true(
            compare_mat(mn_iw_modeld.iw->cov_scale_posterior,
                        posterior_cov_scale, 1e-10));

        // Posterior correcly attached to row_cov
        set_r_seed(2);

        set_r_seed(2);
        mn_iw_modeld.sample_posterior();
        // mn_iw_modeld.iw->cov.raw_print(Rcpp::Rcerr);

        arma::mat posterior_cov_sample = {
            {0.497339646363078, -0.236486515430326},
            {-0.236486515430326, 0.883150965865179}};

        expect_true(
            compare_mat(mn_iw_modeld.iw->cov,
                        posterior_cov_sample, 1e-10));

        // log_likelihood

        // Rcpp::Rcerr << mn_iw_modeld.log_marginal_likelihood() << std::endl;

        expect_true(
            compare_double(mn_iw_modeld.log_marginal_likelihood(),
                           -175.583325284281,
                           1e-10));
    };
};
