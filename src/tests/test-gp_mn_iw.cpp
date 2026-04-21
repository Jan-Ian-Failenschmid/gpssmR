// [[Rcpp::depends(RcppArmadillo)]]

#include <testthat.h>
#include <RcppArmadillo.h>
#include "test_helper.h"
#include "linear_algebra.h"
#include "imc_gp_struct.h"
#include "main_helper.h"

context("C++ GP Matrix-normal-inverse-Wishart")
{
    test_that("GP matrix-normal-inverse-Wishart matches reference values")
    {
        const double tol = 1e-10;

        set_r_seed(1);
        const arma::uword n = 50;
        const arma::uword d1 = 2;
        const arma::uword d2 = 2;

        arma::mat X(d1, n, arma::fill::randn);
        arma::mat covariate(d2, n, arma::fill::randn);

        arma::mat covar_mat(d1, d2, arma::fill::randn);
        arma::mat covar_mat_mean(d1, d2, arma::fill::randn);

        const arma::mat covar_prior_col_cov = identity(d2);
        const arma::mat cov = 0.5 * identity(d1);
        const arma::uword cov_df = 4;
        const arma::mat cov_scale = identity(d1);

        arma::mat Y(d1, n, arma::fill::randn);
        Y = arma::chol(cov, "lower") * Y;
        Y += arma::tanh(X) + covar_mat * covariate;

        arma::mat dyn_mat_mean(d1, n, arma::fill::zeros);
        arma::mat data_mean = Y;
        data_mean.zeros();
        arma::mat covar_col_cov_chol = chol(covar_prior_col_cov, "lower");
        arma::mat cov_scale_chol = chol(cov_scale, "lower");
        arma::mat data_cov = identity(n);

        auto gp = std::make_unique<imc_gp>();
        gp->set_hyperparameters(5.0, 1.0);
        gp->update_predictor(X);

        mn_covar_wrapper model_wrapper(
            gp->get_predictor_ptr(), &covariate,
            &dyn_mat_mean, &covar_mat_mean,
            gp->get_cov_chol_ptr(), &covar_col_cov_chol);

        mn_iw_model_ model = init_mn_iw_model(
            Y,
            data_mean,
            data_cov,
            model_wrapper,
            cov_scale_chol,
            cov_df);

        model.calc_posterior_parameters();
        const arma::mat posterior_cov_scale = {
            {23.0381917797606, -1.42030578326805},
            {-1.42030578326805, 27.6275728158056}};
        expect_true(
            compare_mat(model.iw->cov_scale_posterior,
                        posterior_cov_scale, tol));

        set_r_seed(2);
        model.sample_posterior();

        const arma::mat posterior_cov_sample = {
            {0.497339646363078, -0.236486515430326},
            {-0.236486515430326, 0.883150965865179}};

        expect_true(
            compare_mat(model.iw->cov,
                        posterior_cov_sample, tol));
        expect_true(
            compare_double(model.log_marginal_likelihood(),
                           -175.583325341299,
                           tol));
    };
};
