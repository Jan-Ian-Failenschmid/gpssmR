// [[Rcpp::depends(RcppArmadillo)]]

#include <testthat.h>
#include <RcppArmadillo.h>
#include "t_helper.h"
#include "linear_algebra.h"
#include "imc_iw_class.h"
#include "imc_gp_class.h"
#include "base_structs.h"
#include "derived_structs.h"

context("C++ GP Matrix-normal-inverse-Wishart")
{
    test_that("GP Matrix-normal-inverse-Wishart")
    {
        // Data generation
        set_r_seed(1);
        arma::uword n = 50;
        arma::uword d = 2;

        arma::mat X(d, n, arma::fill::randn);
        arma::mat covariate(d, n, arma::fill::randn);

        arma::vec hyperparameters = {5.0, 1.0};

        arma::mat covar_mat(d, d, arma::fill::randn);

        arma::mat covar_mat_const(d, d, arma::fill::zeros);
        covar_mat_const.fill(arma::datum::nan);

        arma::mat covar_mat_mean(d, d, arma::fill::randn);

        arma::mat covar_prior_col_cov = identity(d);

        arma::mat cov = 0.5 * identity(d);

        arma::uword cov_df = 4;
        arma::mat cov_scale = identity(d);

        arma::mat Y(d, n, arma::fill::randn);
        Y = arma::chol(cov, "lower") * Y;
        Y += arma::tanh(X) + covar_mat * covariate;

        Dataset data;
        data.outcome = &Y;
        data.predictors = {&X, &covariate};
        arma::mat data_mean = Y;
        data_mean.zeros();

        // Computation
        // Imc_iw old
        imc_iw matniw_gp_model(
            hyperparameters,
            covar_mat_mean,
            covar_prior_col_cov,
            cov_df,
            cov_scale);
        matniw_gp_model.update_data(Y, X, covariate);
        matniw_gp_model.calc_posterior_parameters();
        // New model

        auto gp1 = std::make_unique<imc_gp>();
        auto mn_gp1 = std::make_unique<mn_gp_mean_model_>(
            std::move(gp1), identity(d), 0);
        mn_gp1->set_hyperparameters(5.0, 1.0);

        auto mn_covar1 = std::make_unique<mn_regression_model_>(
            covar_mat_mean, covar_prior_col_cov, identity(d), 1);

        auto gp_covar1 = std::make_unique<mn_mn_model_>(
            std::move(mn_gp1), std::move(mn_covar1));
        auto iw1 = std::make_unique<iw_model_>(cov_df, cov_scale);

        mn_iw_model_ gp_mn_iw1(std::move(gp_covar1), std::move(iw1));

        gp_mn_iw1.set_data(data, data_mean, identity(n));
        gp_mn_iw1.calc_posterior_parameters();

        auto mn_chain1 = dynamic_cast<mn_mn_model_ *>(gp_mn_iw1.mn.get());

        // New model reverse order
        auto gp2 = std::make_unique<imc_gp>();
        auto mn_gp2 = std::make_unique<mn_gp_mean_model_>(
            std::move(gp2), identity(d), 0);
        mn_gp2->set_hyperparameters(5.0, 1.0);

        auto mn_covar2 = std::make_unique<mn_regression_model_>(
            covar_mat_mean, covar_prior_col_cov, identity(d), 1);

        auto gp_covar2 = std::make_unique<mn_mn_model_>(
            std::move(mn_covar2), std::move(mn_gp2));
        auto iw2 = std::make_unique<iw_model_>(cov_df, cov_scale);

        mn_iw_model_ gp_mn_iw2(std::move(gp_covar2), std::move(iw2));

        gp_mn_iw2.set_data(data, data_mean, identity(n));
        gp_mn_iw2.calc_posterior_parameters();

        auto mn_chain2 = dynamic_cast<mn_mn_model_ *>(gp_mn_iw2.mn.get());

        // Expectation
        Rcpp::Rcerr.precision(15);

        // Posterior covariate mean
        // matniw_gp_model.covar_mean_post.raw_print(Rcpp::Rcerr);
        // mn_chain1->mn2->param_posterior.raw_print(Rcpp::Rcerr);
        arma::mat posterior_covar_mean = {
            {-1.32343807656027, -1.04405345854874},
            {1.30774788302267, 1.52516728205828}};

        expect_true(
            compare_mat(mn_chain1->mn2->param_posterior,
                        posterior_covar_mean, 1e-10));

        // Posterior covariate col
        // matniw_gp_model.covar_col_cov_post.raw_print(Rcpp::Rcerr);
        // mn_chain1->mn2->col_cov_posterior.raw_print(Rcpp::Rcerr);
        arma::mat posterior_covar_col_cov = {
            {0.0316024819136566, -0.00521360092988983},
            {-0.00521360092988983, 0.0426756951477309}};

        expect_true(
            compare_mat(mn_chain1->mn2->col_cov_posterior,
                        posterior_covar_col_cov, 1e-10));

        // Posterior cov scale
        // matniw_gp_model.post_cov_scale.raw_print(Rcpp::Rcerr);
        // gp_mn_iw1.iw->cov_scale_posterior.raw_print(Rcpp::Rcerr);
        arma::mat posterior_cov_scale = {
            {23.0381917812906, -1.42030578276968},
            {-1.42030578276968, 27.627572816381}};

        expect_true(
            compare_mat(gp_mn_iw1.iw->cov_scale_posterior,
                        posterior_cov_scale, 1e-10));

        // Posterior correcly attached to row_cov
        set_r_seed(2);
        matniw_gp_model.sample_joint_posterior();
        // matniw_gp_model.cov.raw_print(Rcpp::Rcerr);
        set_r_seed(2);
        gp_mn_iw1.sample_posterior();
        // mn_chain1->mn1->row_cov.raw_print(Rcpp::Rcerr);

        arma::mat posterior_cov_sample = {
            {0.497339646363078, -0.236486515430326},
            {-0.236486515430326, 0.883150965865179}};

        expect_true(
            compare_mat(mn_chain1->mn1->row_cov,
                        posterior_cov_sample, 1e-10));

        // log_likelihood
        // Rcpp::Rcerr << matniw_gp_model.log_marginal_likelihood() << std::endl;
        // Rcpp::Rcerr << gp_mn_iw1.log_marginal_likelihood() << std::endl;
        // Rcpp::Rcerr << gp_mn_iw2.log_marginal_likelihood() << std::endl;

        expect_true(
            compare_double(gp_mn_iw1.log_marginal_likelihood(),
                           -175.583325242305,
                           1e-10));

        expect_true(
            compare_double(gp_mn_iw1.log_marginal_likelihood(),
                           gp_mn_iw2.log_marginal_likelihood(),
                           1e-10));
    };
};