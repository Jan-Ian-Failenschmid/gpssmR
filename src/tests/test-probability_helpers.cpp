// [[Rcpp::depends(RcppArmadillo)]]

#include <testthat.h>
#include <RcppArmadillo.h>
#include "test_helper.h"
#include "linear_algebra.h"
#include "pdfs.h"

context("C++ Probability helpers")
{
    test_that("Log-density helpers agree with direct calculations")
    {
        const double tol = 1e-10;

        arma::vec x = {1.0, -0.5};
        arma::vec mu = {0.25, 0.75};
        arma::mat cov = {
            {2.0, 0.3},
            {0.3, 1.5}};
        arma::mat chol_cov = arma::chol(cov, "lower");
        arma::vec diff = x - mu;
        arma::vec z = arma::solve(arma::trimatl(chol_cov), diff,
                                  arma::solve_opts::fast);
        double expected_logdnorm =
            -0.5 * chol_cov.n_cols * std::log(2.0 * M_PI) -
            0.5 * log_det_chol(chol_cov) -
            0.5 * arma::dot(z, z);

        expect_true(compare_double(
            logdnorm(x, mu, chol_cov), expected_logdnorm, tol));

        arma::mat X = {
            {1.0, 0.0},
            {-0.5, 0.5}};
        arma::mat Mu = {
            {0.25, 1.0},
            {0.75, -0.25}};
        arma::vec expected_mat_logdnorm(2);
        arma::vec expected_unnorm(2);
        arma::mat inv_cov = arma::inv_sympd(cov);
        for (arma::uword i = 0; i < X.n_cols; i++)
        {
            expected_mat_logdnorm(i) = logdnorm(X.col(i), Mu.col(i), chol_cov);
            arma::vec col_diff = X.col(i) - Mu.col(i);
            expected_unnorm(i) = -0.5 * arma::dot(col_diff, inv_cov * col_diff);
        }

        expect_true(compare_mat(
            mat_logdnorm(X, Mu, chol_cov), expected_mat_logdnorm, tol));
        expect_true(compare_mat(
            mat_logdnorm(X, mu, chol_cov),
            arma::vec({
                logdnorm(X.col(0), mu, chol_cov),
                logdnorm(X.col(1), mu, chol_cov)}),
            tol));
        expect_true(compare_mat(
            mat_logdnorm(x, Mu, chol_cov),
            arma::vec({
                logdnorm(x, Mu.col(0), chol_cov),
                logdnorm(x, Mu.col(1), chol_cov)}),
            tol));

        expect_true(compare_mat(
            mat_logdnorm_unnorm_inv(X, Mu, inv_cov),
            expected_unnorm, tol));
        expect_true(compare_mat(
            mat_logdnorm_unnorm_inv(X, mu, inv_cov),
            arma::vec({
                -0.5 * arma::dot(X.col(0) - mu, inv_cov * (X.col(0) - mu)),
                -0.5 * arma::dot(X.col(1) - mu, inv_cov * (X.col(1) - mu))}),
            tol));
        expect_true(compare_mat(
            mat_logdnorm_unnorm_inv(x, Mu, inv_cov),
            arma::vec({
                -0.5 * arma::dot(x - Mu.col(0), inv_cov * (x - Mu.col(0))),
                -0.5 * arma::dot(x - Mu.col(1), inv_cov * (x - Mu.col(1)))}),
            tol));
    }

    test_that("Probability helper summaries match direct formulas")
    {
        const double tol = 1e-10;

        arma::vec cond_mu;
        arma::vec cond_sigma;
        arma::vec mu = {1.0, 2.0, 3.0};
        arma::mat sigma = {
            {2.0, 0.2, 0.1},
            {0.2, 1.5, 0.3},
            {0.1, 0.3, 1.2}};
        arma::vec fixed_vals = {1.5, 2.5};
        arma::uvec fixed = {0, 2};
        arma::uvec free = {1};
        conditional_mvnormal(cond_mu, cond_sigma, mu, sigma,
                             fixed_vals, fixed, free);

        arma::vec mu_free = mu(free);
        arma::vec mu_fixed = mu(fixed);
        arma::mat sigma_free = sigma(free, free);
        arma::mat sigma_free_fixed = sigma(free, fixed);
        arma::mat sigma_fixed_free = sigma(fixed, free);
        arma::mat sigma_fixed = sigma(fixed, fixed);
        arma::mat sigma_fixed_inv = arma::pinv(sigma_fixed);

        arma::vec expected_cond_mu =
            mu_free + sigma_free_fixed * sigma_fixed_inv * (fixed_vals - mu_fixed);
        arma::vec expected_cond_sigma =
            arma::vec({arma::as_scalar(
                sigma_free - sigma_free_fixed * sigma_fixed_inv * sigma_fixed_free)});

        expect_true(compare_mat(cond_mu, expected_cond_mu, tol));
        expect_true(compare_mat(cond_sigma, expected_cond_sigma, tol));

        arma::mat row_chol = {
            {1.2, 0.0},
            {0.3, 1.1}};
        arma::mat col_chol = {
            {1.1, 0.0},
            {-0.2, 0.9}};
        arma::mat mean = {
            {0.5, -1.0},
            {1.5, 0.25}};
        arma::mat obs = {
            {0.8, -0.7},
            {1.2, 0.5}};

        arma::mat diff = obs - mean;
        arma::mat Z = arma::solve(arma::trimatl(row_chol), diff,
                                  arma::solve_opts::fast);
        arma::mat W = arma::solve(arma::trimatl(col_chol), Z.t(),
                                  arma::solve_opts::fast);
        double expected_logdmatnorm =
            -0.5 * arma::accu(W % W) -
            0.5 * (row_chol.n_rows * col_chol.n_rows * std::log(2.0 * M_PI) +
                   col_chol.n_rows * log_det_chol(row_chol) +
                   row_chol.n_rows * log_det_chol(col_chol));

        expect_true(compare_double(
            logdmatnorm(obs, mean, row_chol, col_chol),
            expected_logdmatnorm,
            tol));

        double expected_mvgamma =
            0.5 * std::log(M_PI) + std::lgamma(3.0) + std::lgamma(2.5);
        expect_true(compare_double(log_mvgamma(3.0, 2), expected_mvgamma, tol));

        arma::mat sigma_y_chol = arma::diagmat(arma::vec({1.0, 2.0}));
        arma::mat prior_cov_scale_chol = arma::diagmat(arma::vec({1.5, 0.75}));
        arma::mat post_cov_scale_chol = arma::diagmat(arma::vec({2.0, 1.25}));
        arma::uword v_prior = 4;
        arma::uword v_post = 7;
        arma::uword dy = prior_cov_scale_chol.n_cols;
        arma::uword n = v_post - v_prior;
        double expected_log_dmatrixt =
            -std::log(M_PI) * (dy * n / 2.0) -
            (dy / 2.0) * log_det_chol(sigma_y_chol) +
            (v_prior / 2.0) * log_det_chol(prior_cov_scale_chol) -
            (v_post / 2.0) * log_det_chol(post_cov_scale_chol) +
            log_mvgamma(v_post / 2.0, dy) -
            log_mvgamma(v_prior / 2.0, dy);
        expect_true(compare_double(
            log_dmatrixt(sigma_y_chol,
                         prior_cov_scale_chol,
                         post_cov_scale_chol,
                         v_prior,
                         v_post),
            expected_log_dmatrixt,
            tol));
    }

    test_that("Random matrix helper draws are reproducible under set_r_seed")
    {
        const double tol = 1e-10;

        arma::mat mean = {
            {0.5, -1.0},
            {1.5, 0.25}};
        arma::mat cov_chol = {
            {1.1, 0.0},
            {-0.3, 0.8}};

        set_r_seed(42);
        arma::mat sample_a = mat_rnorm(mean, cov_chol);
        set_r_seed(42);
        arma::mat sample_b = mat_rnorm(mean, cov_chol);
        expect_true(compare_mat(sample_a, sample_b, tol));

        arma::mat out_a(2, 2);
        arma::mat out_b(2, 2);
        set_r_seed(99);
        mat_rnorm(out_a, mean, cov_chol);
        set_r_seed(99);
        mat_rnorm(out_b, mean, cov_chol);
        expect_true(compare_mat(out_a, out_b, tol));

        arma::mat row_chol = {
            {1.2, 0.0},
            {0.3, 1.1}};
        arma::mat col_chol = {
            {1.1, 0.0},
            {-0.2, 0.9}};

        set_r_seed(7);
        arma::mat rmat_a = rmatnorm(mean, row_chol, col_chol);
        set_r_seed(7);
        arma::mat rmat_b = rmatnorm(mean, row_chol, col_chol);
        expect_true(compare_mat(rmat_a, rmat_b, tol));
    }
}
