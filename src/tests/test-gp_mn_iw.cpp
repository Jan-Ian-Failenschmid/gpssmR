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

        arma::mat identity_pred = identity(n);

        Dataset data;
        data.outcome = &Y;
        data.predictors = {&X, &covariate, &identity_pred};
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
            std::move(gp1), identity(d1), 0);
        mn_gp1->set_hyperparameters(5.0, 1.0);

        auto mn_covar1 = std::make_unique<mn_regression_model_>(
            covar_mat_mean, covar_prior_col_cov, identity(d1), 1);

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
            std::move(gp2), identity(d1), 0);
        mn_gp2->set_hyperparameters(5.0, 1.0);

        auto mn_covar2 = std::make_unique<mn_regression_model_>(
            covar_mat_mean, covar_prior_col_cov, identity(d1), 1);

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

        mn_covar_wrapper mn_wrapper(
            &identity_pred, &covariate,
            mn_chain1->mn1->param_prior, covar_mat_mean, mn_chain1->mn1->col_cov_prior, covar_prior_col_cov);
        data.predictors = {&X, &covariate, &mn_wrapper.combined_data};

        mn_regression_model_ mn_mod(mn_wrapper.get_mean(),
                                    mn_wrapper.get_cov(),
                                    gp_mn_iw1.iw->get_cov(), 2);

        mn_mod.set_data(data, data_mean, identity(n));
        mn_mod.calc_posterior_parameters();

        Rcpp::Rcerr << mn_mod.param_posterior << std::endl;
        Rcpp::Rcerr << mn_mod.col_cov_posterior << std::endl;

        Rcpp::Rcerr << mn_mod.param_prior << std::endl;
        Rcpp::Rcerr << mn_mod.col_cov_prior << std::endl;

        Rcpp::Rcerr << matniw_gp_model.des_covar_mat << std::endl;
        Rcpp::Rcerr << matniw_gp_model.covar_col_cov_post << std::endl;
        Rcpp::Rcerr << matniw_gp_model.covar_mean_post << std::endl;

        matniw_gp_model.update_data(Y, X, covariate);
        matniw_gp_model.make_gp_predictions();

        Rcpp::Rcerr
            << matniw_gp_model.multiv_gp.pred_col_cov_chol *
                   matniw_gp_model.multiv_gp.pred_col_cov_chol.t()
            << std::endl;

        arma::mat k_chol = matniw_gp_model.multiv_gp.train_k_chol;

        // Construct "augmented" K including covariates in prior
        arma::mat K = k_chol * k_chol.t();

        arma::mat K_aug = K + covariate.t() * covar_prior_col_cov * covariate;

        // GP marginal posterior covariance
        arma::mat Cov_gp_marginal_alt = K - K * arma::inv_sympd(K_aug + identity_pred) * K;

        Rcpp::Rcerr << "GP marginal posterior via predictive formula:\n"
                    << Cov_gp_marginal_alt << std::endl;

        // Suppose mn_mod.col_cov_posterior is the covariance
        arma::mat Cov_joint = mn_mod.col_cov_posterior;
        arma::mat P_joint = Cov_joint;

        // Define blocks (assuming GP coefficients come first)
        arma::mat A = P_joint.submat(0, 0, n - 1, n - 1);
        arma::mat B = P_joint.submat(0, n, n - 1, n + d2 - 1);
        arma::mat D = P_joint.submat(n, n, n + d2 - 1, n + d2 - 1);

        // GP marginal precision
        arma::mat P_gp = A - B * arma::inv_sympd(D) * B.t();

        // GP marginal covariance
        arma::mat Cov_gp = arma::inv_sympd(P_gp);

        Rcpp::Rcerr
            << P_gp
            << std::endl;

        Rcpp::Rcerr
            << arma::inv_sympd(D - B.t() * arma::inv_sympd(A) * B)
            << std::endl;

        // blocks
        arma::mat Sigma_gp = Cov_joint.submat(0, 0, n - 1, n - 1);
        arma::mat Sigma_gx = Cov_joint.submat(0, n, n - 1, n + d2 - 1);
        arma::mat Sigma_xg = Cov_joint.submat(n, 0, n + d2 - 1, n - 1);
        arma::mat Sigma_xx = Cov_joint.submat(n, n, n + d2 - 1, n + d2 - 1);

        // conditional covariance
        arma::mat Cov_gp_cond =
            Sigma_gp - Sigma_gx * arma::inv_sympd(Sigma_xx) * Sigma_xg;

        Rcpp::Rcerr << "GP conditional covariance:\n"
                    << Cov_gp_cond << std::endl;

        // GP marginal posterior covariance

        Rcpp::Rcerr << K - K * arma::inv_sympd(K + identity_pred) * K
        << std::endl;
    };
};