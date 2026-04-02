// [[Rcpp::depends(RcppArmadillo)]]

#include <testthat.h>
#include <RcppArmadillo.h>
#include "test_helper.h"
#include "linear_algebra.h"
#include "main_helper.h"

context("C++ Multivariate-normal-inverse-Wishart")
{
    test_that("Multivariate-normal-inverse-Wishart")
    {
        // Data generation
        set_r_seed(1);
        arma::uword n = 50;
        arma::uword d1 = 2;
        arma::uword d2 = 4;

        arma::mat X(d1, n, arma::fill::randn);
        arma::mat covariate(0, n, arma::fill::zeros);

        arma::mat des_mat(d2, d1);
        arma::mat covar_mat(d2, 0, arma::fill::zeros);

        arma::mat des_mat_const(d2, d1, arma::fill::zeros);
        des_mat_const.fill(arma::datum::nan);
        des_mat_const(0, 0) = 1.0;
        des_mat_const(1, 1) = 0.0;
        des_mat_const(2, 1) = 1.0;
        des_mat_const(3, 0) = 0.0;

        arma::mat covar_mat_const(d2, 0, arma::fill::zeros);
        covar_mat_const.fill(arma::datum::nan);

        arma::vec des_mat_mean(4, arma::fill::randn);
        arma::vec covar_mat_mean(0, arma::fill::zeros);

        arma::mat prior_cov = identity(4);
        arma::mat covar_prior_cov = identity(0);

        arma::vec des_mat_sample(4);
        mat_rnorm(des_mat_sample, des_mat_mean, arma::chol(prior_cov, "lower"));
        des_mat = vec2mat(des_mat_sample, des_mat_const);

        arma::mat cov = identity(d2);

        arma::uword cov_df = 4;
        arma::mat cov_scale = identity(d2);

        arma::mat Y(d2, n, arma::fill::randn);
        Y = arma::chol(cov, "lower") * Y;
        Y += des_mat * X;

        // Dataset data;
        // data.outcome = &Y;
        // data.predictors = {&X, &covariate};

        // Computation
        arma::mat data_mean = Y;
        data_mean.zeros();

        mvn_covar_wrapper model_wrapper(
            &X,
            &covariate,
            &des_mat_const,
            &covar_mat_const,
            &des_mat_mean,
            &covar_mat_mean,
            &prior_cov,
            &covar_prior_cov);

        mvn_iw_model_ multniw_model2 = init_mvn_iw_model(
            Y,
            data_mean,
            model_wrapper,
            cov_scale,
            cov_df);
            
        set_r_seed(2);
        multniw_model2.sample_prior();
        multniw_model2.calc_posterior_parameters();

        // Expectations
        // Prior cov sample
        Rcpp::Rcerr.precision(15);
        // multniw_model2.iw->get_cov().raw_print(Rcpp::Rcerr);
        // multniw_model2.mvn->cov_prior.raw_print(Rcpp::Rcerr);

        arma::mat prior_cov_sample = {
            {1.12285022850069, 0.422712962586741,
             -0.702298892575532, -0.683013033610507},
            {0.422712962586741, 2.52222863653428,
             -1.01832931328674, -2.00317240910251},
            {-0.702298892575532, -1.01832931328674,
             0.949095566763763, 0.95389420212767},
            {-0.683013033610507, -2.00317240910251,
             0.95389420212767, 1.82166520810286}};

        expect_true(
            compare_mat(multniw_model2.get_cov(), prior_cov_sample, 1e-10));

        // Posterior mean
        // multniw_model2.mvn->mean_posterior.raw_print(Rcpp::Rcerr);
        arma::vec posterior_mean = {
            1.52883767295901, -0.277212425771659, 0.0980639933164831,
            -2.87072565287718};

        expect_true(
            compare_mat(multniw_model2.mvn->mean_posterior,
                        posterior_mean, 1e-10));

        // Posterior cov

        // multniw_model2.mvn->cov_posterior.raw_print(Rcpp::Rcerr);
        arma::mat posterior_covariance = {
            {0.00448136954778958, -0.00228235275119866, 1.53683861127445e-05, -4.72289835398757e-05},
            {-0.00228235275119866, 0.00716353876781744, -8.33712778975189e-05, 4.45471280415266e-05},
            {1.53683861127445e-05, -8.33712778975189e-05, 0.00919183968423502, -0.00355134888268583},
            {-4.72289835398757e-05, 4.45471280415266e-05, -0.00355134888268583, 0.00335567114202137}};

        expect_true(compare_mat(multniw_model2.mvn->cov_posterior,
                        posterior_covariance, 1e-10));

        // Posterior sample_mean
        set_r_seed(3);
        multniw_model2.sample_posterior();

        // model_wrapper.get_pred_param().raw_print(Rcpp::Rcerr);
        arma::mat posterior_param_sample = {
            {1.0, -0.00242569121853596},
            {1.49780861793135, 0.0},
            {-0.228146144338885, 1},
            {0, -2.90118695077391}};

        expect_true(
            compare_mat(model_wrapper.get_pred_param(),
                        posterior_param_sample, 1e-10));

        // Posterior sample_cov
        // multniw_model2.iw->get_cov().raw_print(Rcpp::Rcerr);
        arma::mat posterior_cov_sample = {
            {0.936093903463749, -0.0744679452097038,
             0.0114717724579501, -0.447879919193225},
            {-0.0744679452097038, 1.0306300522827,
             -0.0314046685747652, -0.241237539878783},
            {0.0114717724579501, -0.0314046685747652,
             1.05270332791306, -0.067900549205874},
            {-0.447879919193225, -0.241237539878783,
             -0.067900549205874, 1.21064036293343}};

        expect_true(
            compare_mat(multniw_model2.get_cov(),
                        posterior_cov_sample, 1e-10));
    };
};
