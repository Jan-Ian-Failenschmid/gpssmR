// [[Rcpp::depends(RcppArmadillo)]]

#include <testthat.h>
#include <RcppArmadillo.h>
#include "t_helper.h"
#include "linear_algebra.h"
#include "matniw_gp_class.h"
#include "hsgp_class.h"
#include "base_structs.h"
#include "derived_structs.h"

context("C++ HSGP Matrix-normal-inverse-Wishart")
{
    test_that("HSGP Matrix-normal-inverse-Wishart")
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

        Dataset data;
        data.outcome = &Y;
        data.predictors = {&X, &covariate};
        arma::mat data_mean = Y;
        data_mean.zeros();

        arma::mat basis_fun_index = expand_grid_2d(arma::regspace(1, 2),
                                                   arma::regspace(1, 2));
        arma::vec boundry_factor = {4, 4};
        arma::mat dyn_mat_const(d1,
                                basis_fun_index.n_rows,
                                arma::fill::zeros);
        dyn_mat_const.fill(arma::datum::nan);
        arma::mat dyn_mat_mean = dyn_mat_const;
        dyn_mat_mean.zeros();

        // Computation
        // Imc_iw old
        matniw_gp_model matniw_gp_model(
            basis_fun_index,
            boundry_factor,
            hyperparameters,
            dyn_mat_const,
            covar_mat_const,
            dyn_mat_mean,
            covar_mat_mean,
            covar_prior_col_cov,
            cov_df,
            cov_scale);

        matniw_gp_model.update_hyperparameters(5.0, 1.0);
        matniw_gp_model.update_data(Y, X, covariate);
        matniw_gp_model.calc_posterior_parameters();

        // New model
        auto gp = std::make_unique<hsgp_approx>(basis_fun_index,
                                                boundry_factor);
        auto mn_gp = std::make_unique<mn_hsgp_regression_model_>(
            std::move(gp), identity(d1), 0);
        mn_gp->set_hyperparameters(5.0, 1.0);

        auto mn_covar = std::make_unique<mn_regression_model_>(
            covar_mat_mean, covar_prior_col_cov, identity(d2), 1);

        auto gp_covar = std::make_unique<mn_mn_model_>(
            std::move(mn_gp), std::move(mn_covar));
        auto iw = std::make_unique<iw_model_>(cov_df, cov_scale);

        mn_iw_model_ gp_mn_iw(std::move(gp_covar), std::move(iw));

        gp_mn_iw.set_data(data, data_mean, identity(n));
        gp_mn_iw.calc_posterior_parameters();

        auto mn_chain = dynamic_cast<mn_mn_model_ *>(gp_mn_iw.mn.get());

        // Expectation
        Rcpp::Rcerr.precision(15);

        // Prior mean mean
        // matniw_gp_model.prior_des_mat_mean.raw_print(Rcpp::Rcerr);
        // mn_chain->mn1->param_prior.raw_print(Rcpp::Rcerr);
        arma::mat mn1_param_prior = {
            {0, 0, 0, 0},
            {0, 0, 0, 0}};
        expect_true(
            compare_mat(mn_chain->mn1->param_prior, mn1_param_prior, 1e-10));

        // mn_chain->mn2->param_prior.raw_print(Rcpp::Rcerr);
        arma::mat mn2_param_prior = {
            {1.85282612544816, -2.21148730568689},
            {0.537033294340683, -1.48572790117537}};
        expect_true(
            compare_mat(mn_chain->mn2->param_prior, mn2_param_prior, 1e-10));

        // Posterior mean
        // matniw_gp_model.post_des_mat_mean.raw_print(Rcpp::Rcerr);
        // mn_chain->mn1->param_posterior.raw_print(Rcpp::Rcerr);
        arma::mat mn1_param_posterior = {
            {-1.04970107969558, -6.45438550993741, 0.100639663019613,
             1.82074340136688},
            {-0.282017962245035, 2.54636735245928, -4.13086302388682,
             -1.40449122182647}};
        expect_true(
            compare_mat(mn_chain->mn1->param_posterior, mn1_param_posterior, 1e-10));

        // mn_chain->mn2->param_posterior.raw_print(Rcpp::Rcerr);
        arma::mat mn2_param_posterior = {
            {-1.34322786038633, -1.02055453340232},
            {1.29935739164591, 1.67527013651317}};
        expect_true(
            compare_mat(mn_chain->mn2->param_posterior, mn2_param_posterior, 1e-10));

        // Prior column covariance
        // matniw_gp_model.prior_col_cov.raw_print(Rcpp::Rcerr);
        // mn_chain->mn1->col_cov_prior.raw_print(Rcpp::Rcerr);
        arma::mat mn1_col_cov_prior = {
            {53.710138861395, 0, 0, 0},
            {0, 42.6182658055272, 0, 0},
            {0, 0, 42.6182658055272, 0},
            {0, 0, 0, 33.8170151627754}};
        expect_true(
            compare_mat(mn_chain->mn1->col_cov_prior, mn1_col_cov_prior, 1e-10));

        // mn_chain->mn2->col_cov_prior.raw_print(Rcpp::Rcerr);
        arma::mat mn2_col_cov_prior = {
            {1, 0},
            {0, 1}};
        expect_true(
            compare_mat(mn_chain->mn2->col_cov_prior, mn2_col_cov_prior, 1e-10));

        // Posterior column covariance
        // matniw_gp_model.post_col_cov.raw_print(Rcpp::Rcerr);
        // mn_chain->mn1->col_cov_posterior.raw_print(Rcpp::Rcerr);
        arma::mat mn1_col_cov_posterior = {
            {0.451003319530464, 0.155305790729594, -0.0543445036932942,
             -0.0826561644925654},
            {0.155305790729594, 1.23436505874544, 0.0347045833781777,
             -0.717520296676825},
            {-0.0543445036932942, 0.0347045833781777, 0.887952879298786,
             -0.231365213793587},
            {-0.0826561644925654, -0.717520296676825, -0.231365213793587,
             2.83910932115674}};
        expect_true(
            compare_mat(mn_chain->mn1->col_cov_posterior,
                        mn1_col_cov_posterior, 1e-10));

        // mn_chain->mn2->col_cov_posterior.raw_print(Rcpp::Rcerr);
        arma::mat mn2_col_cov_posterior = {
            {0.0211960271011452, -0.00298216568218849},
            {-0.00298216568218849, 0.032496525906481}};
        expect_true(
            compare_mat(mn_chain->mn2->col_cov_posterior,
                        mn2_col_cov_posterior, 1e-10));

        // set_r_seed(2);
        // matniw_gp_model.sample_joint_posterior();
        // matniw_gp_model.cov.raw_print(Rcpp::Rcerr);
        // matniw_gp_model.des_mat.raw_print(Rcpp::Rcerr);

        // Posterior sample
        set_r_seed(2);
        gp_mn_iw.sample_posterior();

        // gp_mn_iw.iw->get_cov().raw_print(Rcpp::Rcerr);
        arma::mat posterior_cov_sample = {
            {0.571660278104027, -0.286236580000498},
            {-0.286236580000498, 1.16750353556818}};
        expect_true(
            compare_mat(gp_mn_iw.iw->get_cov(), posterior_cov_sample, 1e-10));

        // mn_chain->mn1->get_param().raw_print(Rcpp::Rcerr);
        arma::mat mn1_param_sample = {
            {-1.44623239584311, -7.59552193449697, -0.956390515436832,
             2.0631024395975},
            {0.446084937028854, 2.90107413711889, -3.96889045982167,
             -0.980318508039314}};
        expect_true(
            compare_mat(mn_chain->mn1->get_param(), mn1_param_sample, 1e-10));

        // mn_chain->mn2->get_param().raw_print(Rcpp::Rcerr);
        arma::mat mn2_param_sample = {
            {-1.2045328204345, -0.781119576094488},
            {1.21210593804881, 1.92314672587511}};
        expect_true(
            compare_mat(mn_chain->mn2->get_param(), mn2_param_sample, 1e-10));

        // Log-likelihood
        // Rcpp::Rcerr << gp_mn_iw.log_marginal_likelihood() << std::endl;
        // Rcpp::Rcerr << matniw_gp_model.log_marginal_likelihood() << std::endl;
        expect_true(
            compare_double(gp_mn_iw.log_marginal_likelihood(),
                           -144.087446069717, 1e-10));
    };
};