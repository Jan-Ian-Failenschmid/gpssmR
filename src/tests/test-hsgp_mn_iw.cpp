// [[Rcpp::depends(RcppArmadillo)]]

#include <testthat.h>
#include <RcppArmadillo.h>
#include "test_helper.h"
#include "linear_algebra.h"
#include "hsgp_struct.h"
#include "main_helper.h"

context("C++ HSGP Matrix-normal-inverse-Wishart")
{
    test_that("HSGP matrix-normal-inverse-Wishart matches reference values")
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
        arma::mat data_mean(d1, n, arma::fill::zeros);

        const arma::mat basis_fun_index = expand_grid_2d(arma::regspace(1, 2),
                                                         arma::regspace(1, 2));
        const arma::vec boundry_factor = {4, 4};
        arma::mat dyn_mat_const(d1,
                                basis_fun_index.n_rows,
                                arma::fill::zeros);
        dyn_mat_const.fill(arma::datum::nan);
        arma::mat dyn_mat_mean = dyn_mat_const;
        dyn_mat_mean.zeros();

        arma::mat covar_col_cov_chol = chol(covar_prior_col_cov, "lower");
        arma::mat cov_scale_chol = chol(cov_scale, "lower");
        arma::mat data_cov = identity(n);

        auto gp = std::make_unique<hsgp_approx>(basis_fun_index,
                                                boundry_factor);
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

        const arma::mat mn_param_prior = {
            {0, 0, 0, 0, 1.85282612544816, -2.21148730568689},
            {0, 0, 0, 0, 0.537033294340683, -1.48572790117537}};
        expect_true(compare_mat(
            *model.mn->coefficient_prior, mn_param_prior, tol));

        const arma::mat mn_param_posterior = {
            {-0.39884362769862, -4.13126859697866, -0.444045263887508,
             0.127788478708301, -1.3421995363054, -1.01862370937414},
            {-0.768485248441927, -0.480529554499412, -4.44588731286972,
             0.546017786978555, 1.29794996469722, 1.67797378297536}};
        expect_true(
            compare_mat(model.mn->coefficient_posterior,
                        mn_param_posterior, tol));

        const arma::mat mn_col_cov_prior = {
            {134.63135270433, 0, 0, 0, 0, 0},
            {0, 106.828150083874, 0, 0, 0, 0},
            {0, 0, 106.828150083874, 0, 0, 0},
            {0, 0, 0, 84.766686370638, 0, 0},
            {0, 0, 0, 0, 1, 0},
            {0, 0, 0, 0, 0, 1}};
        expect_true(compare_mat(model.mn->col_cov_prior,
                                mn_col_cov_prior, tol));

        const arma::mat mn_col_cov_posterior = {
            {0.465937772177563, 0.166423726790762, -0.0842747734586196,
             -0.102596151698875, -0.014934132680215, 0.00930082690049035},
            {0.166423726790762, 1.31777345485517, 0.0482222950669017,
             -0.80174488135983, -0.0195533009634772, -0.0299156756123358},
            {-0.0842747734586196, 0.0482222950669017, 0.984689618704573,
             -0.227234895200749, 0.0301726668732114, -0.0407568565118588},
            {-0.102596151698875, -0.80174488135983, -0.227234895200749,
             3.02752153632888, 0.0228400893308101, 0.00785541380286091},
            {-0.014934132680215, -0.0195533009634772, 0.0301726668732114,
             0.0228400893308101, 0.0212251452680207, -0.00298987697852006},
            {0.00930082690049035, -0.0299156756123358, -0.0407568565118588,
             0.00785541380286091, -0.00298987697852006, 0.0325338935789517}};

        expect_true(compare_mat(model.mn->col_cov_posterior,
                                mn_col_cov_posterior, tol));

        set_r_seed(2);
        model.sample_posterior();

        const arma::mat posterior_cov_sample = {
            {0.567486334910915, -0.286084572113365},
            {-0.286084572113365, 1.15825209030833}};
        expect_true(compare_mat(model.iw->get_cov(),
                                posterior_cov_sample, tol));

        const arma::mat mn_param_sample = {
            {0.249053944119918, -2.28392066251158, -1.03843058244224,
             -2.33475595185575, -1.56573461617203, -1.07557129570203},
            {-1.17817322336843, 0.834853096798719, -3.21893516963616,
             -0.31425055365385, 1.36391182975267, 1.66154682171166}};
        expect_true(
            compare_mat(model.mn->get_coefficient(),
                        mn_param_sample, tol));

        expect_true(
            compare_double(model.log_marginal_likelihood(),
                           -147.206713133451, tol));
    };
};
