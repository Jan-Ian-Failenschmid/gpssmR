// [[Rcpp::depends(RcppArmadillo)]]

#include <testthat.h>
#include <RcppArmadillo.h>
#include "test_helper.h"
#include "linear_algebra.h"
#include "hsgp_struct.h"
#include "main_helper.h"

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

        // Dataset data;
        // data.outcome = &Y;
        // data.predictors = {&X, &covariate};
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
        // New model
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

        // Prior mean mean
        // mn_iw_modeld.mn->coefficient_prior->raw_print(Rcpp::Rcerr);
        arma::mat mn_param_prior = {
            {0, 0, 0, 0, 1.85282612544816, -2.21148730568689},
            {0, 0, 0, 0, 0.537033294340683, -1.48572790117537}};
        expect_true(compare_mat(
            *mn_iw_modeld.mn->coefficient_prior, mn_param_prior, 1e-10));

        // Posterior mean
        // mn_iw_modeld.mn->coefficient_posterior.raw_print(Rcpp::Rcerr);

        arma::mat mn_param_posterior = {
            {-0.387686081796617, -4.05382709706611, -0.435301686624129,
             0.0760200454194323, -1.34322786038633, -1.02055453340232},
            {-0.767701480134862, -0.460450017401488, -4.38340870288461,
             0.49902910226098, 1.29935739164591, 1.67527013651317},
            {-0.387686081796617, -4.05382709706611, -0.435301686624129,
             0.0760200454194323, -1.34322786038633, -1.02055453340232},
            {-0.767701480134862, -0.460450017401488, -4.38340870288461,
             0.49902910226098, 1.29935739164591, 1.67527013651317}};
        expect_true(
            compare_mat(mn_iw_modeld.mn->coefficient_posterior,
                        mn_param_posterior, 1e-10));

        // Prior column covariance
        // mn_iw_modeld.mn->col_cov_prior.raw_print(Rcpp::Rcerr);
        arma::mat mn_col_cov_prior = {
            {53.710138861395, 0, 0, 0, 0, 0},
            {0, 42.6182658055272, 0, 0, 0, 0},
            {0, 0, 42.6182658055272, 0, 0, 0},
            {0, 0, 0, 33.8170151627754, 0, 0},
            {0, 0, 0, 0, 1, 0},
            {0, 0, 0, 0, 0, 1}};
        expect_true(compare_mat(mn_iw_modeld.mn->col_cov_prior,
                                mn_col_cov_prior, 1e-10));

        // Posterior column covariance
        // Rcpp::Rcerr << std::endl;
        // mn_iw_modeld.mn->col_cov_posterior.raw_print(Rcpp::Rcerr);
        arma::mat mn_col_cov_posterior = {
            {0.462872976105101, 0.161262080832826, -0.0831779947227838,
             -0.0953756265098216, -0.014738191391359, 0.00928619272778908},
            {0.161262080832826, 1.28295860761761, 0.0438766004185812,
             -0.746730359324418, -0.018888085444831, -0.0292616884211955},
            {-0.0831779947227838, 0.0438766004185812, 0.970249677058374,
             -0.212300811992184, 0.0298441761325593, -0.0401423404429411},
            {-0.0953756265098216, -0.746730359324418, -0.212300811992184,
             2.86416561871458, 0.0215458436555165, 0.00702835956857458},
            {-0.014738191391359, -0.018888085444831, 0.0298441761325593,
             0.0215458436555165, 0.0211960271011451, -0.00298216568218847},
            {0.00928619272778908, -0.0292616884211955, -0.0401423404429411,
             0.00702835956857458, -0.00298216568218847, 0.032496525906481}};

        expect_true(compare_mat(mn_iw_modeld.mn->col_cov_posterior,
                                mn_col_cov_posterior, 1e-10));

        // Posterior sample
        set_r_seed(2);
        mn_iw_modeld.sample_posterior();

        // mn_iw_modeld.iw->get_cov().raw_print(Rcpp::Rcerr);
        arma::mat posterior_cov_sample = {
            {0.571660278104027, -0.286236580000498},
            {-0.286236580000498, 1.16750353556818}};
        expect_true(compare_mat(mn_iw_modeld.iw->get_cov(),
                                posterior_cov_sample, 1e-10));

        // mn_iw_modeld.mn->get_coefficient().raw_print(Rcpp::Rcerr);
        arma::mat mn_param_sample = {
            {0.260447628084923, -2.22662202934371, -1.03246859566734,
             -2.30563536164078, -1.56670705884113, -1.07761559187122},
            {-1.17543493860667, 0.854509087399983, -3.16633658620209,
             -0.324330946520751, 1.36476290880426, 1.65914165451348}};
        expect_true(
            compare_mat(mn_iw_modeld.mn->get_coefficient(),
                        mn_param_sample, 1e-10));

        // Log-likelihood
        // Rcpp::Rcerr << gp_mn_iw.log_marginal_likelihood() << std::endl;
        expect_true(
            compare_double(mn_iw_modeld.log_marginal_likelihood(),
                           -144.087446069717, 1e-10));
    };
};
