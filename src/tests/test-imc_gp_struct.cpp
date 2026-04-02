// [[Rcpp::depends(RcppArmadillo)]]

#include <testthat.h>
#include <RcppArmadillo.h>
#include "test_helper.h"
#include "linear_algebra.h"
#include "imc_gp_struct.h"

context("C++ IMC GP struct")
{
    test_that("Incremental IMC GP updates match full recomputation")
    {
        const double tol = 1e-10;

        arma::mat x_train_full(1, 3);
        x_train_full.row(0) = arma::rowvec({0.0, 1.0, 2.0});
        arma::mat y_train_full(1, 3);
        y_train_full.row(0) = arma::rowvec({1.0, -0.5, 0.25});
        arma::mat x_test_full(1, 2);
        x_test_full.row(0) = arma::rowvec({0.5, 1.5});
        arma::mat sigma = identity(1);

        imc_gp gp_full;
        gp_full.update_train_data(x_train_full, y_train_full);
        gp_full.update_sigma(sigma);
        gp_full.set_train_y_cov_I();
        gp_full.update_test_data(x_test_full);
        gp_full.set_test_y_cov_I();
        gp_full.update_hyperparameters(1.5, 0.75);
        gp_full.compute_predictive(true);

        imc_gp gp_seq;
        gp_seq.update_train_data(x_train_full.cols(0, 0), y_train_full.cols(0, 0));
        gp_seq.update_sigma(sigma);
        gp_seq.set_train_y_cov_I();
        gp_seq.append_train_data(x_train_full.cols(1, 2), y_train_full.cols(1, 2));
        gp_seq.append_train_y_cov_I();
        gp_seq.update_test_data(x_test_full.cols(0, 0));
        gp_seq.set_test_y_cov_I();
        gp_seq.append_test_data(x_test_full.cols(1, 1));
        gp_seq.append_test_y_cov_I();
        gp_seq.update_hyperparameters(1.5, 0.75);
        gp_seq.compute_predictive(true);

        expect_true(compare_mat(gp_seq.train_dat, gp_full.train_dat, tol));
        expect_true(compare_mat(gp_seq.outcome_dat, gp_full.outcome_dat, tol));
        expect_true(compare_mat(gp_seq.test_dat, gp_full.test_dat, tol));
        expect_true(compare_mat(gp_seq.train_mu, gp_full.train_mu, tol));
        expect_true(compare_mat(gp_seq.test_mu, gp_full.test_mu, tol));
        expect_true(compare_mat(gp_seq.train_k_chol, gp_full.train_k_chol, 1e-8));
        expect_true(compare_mat(gp_seq.test_k_chol, gp_full.test_k_chol, 1e-8));
        expect_true(compare_mat(gp_seq.train_y_cov_chol, gp_full.train_y_cov_chol, tol));
        expect_true(compare_mat(gp_seq.test_y_cov_chol, gp_full.test_y_cov_chol, tol));
        expect_true(compare_mat(gp_seq.pred_mean, gp_full.pred_mean, 1e-8));
        expect_true(compare_mat(gp_seq.pred_col_cov_chol, gp_full.pred_col_cov_chol, 1e-8));

        expect_true(compare_mat(
            gp_full.kernel(x_train_full),
            gp_full.kernel(x_train_full, x_train_full),
            tol));
        expect_true(compare_mat(
            gp_full.mu(x_test_full),
            arma::zeros(1, x_test_full.n_cols),
            tol));
        expect_true(compare_mat(
            gp_full.get_marginal_train_cov_chol() *
                gp_full.get_marginal_train_cov_chol().t(),
            gp_full.train_k_chol * gp_full.train_k_chol.t() +
                gp_full.train_y_cov_chol * gp_full.train_y_cov_chol.t(),
            1e-8));

        gp_seq.reset_test_data();
        gp_seq.reset_test_y_cov();
        expect_true(compare_double(gp_seq.n_test, 1.0, tol));
        expect_true(compare_mat(gp_seq.test_dat, x_test_full.cols(0, 0), tol));
    }
}
