// [[Rcpp::depends(RcppArmadillo)]]

#include <testthat.h>
#include <RcppArmadillo.h>
#include "test_helper.h"
#include "linear_algebra.h"

context("C++ Linear algebra helpers")
{
    test_that("Linear algebra reshaping helpers match expected outputs")
    {
        const double tol = 1e-10;
        const double nan = arma::datum::nan;

        arma::vec values = {10.0, 20.0, 30.0};
        arma::mat mask = {
            {nan, 1.0, nan},
            {2.0, nan, 3.0}};
        arma::mat expected_matrix = {
            {10.0, 1.0, 30.0},
            {2.0, 20.0, 3.0}};

        expect_true(compare_mat(vec2mat(values, mask), expected_matrix, tol));
        expect_true(compare_mat(mat2vec(expected_matrix, mask), values, tol));

        arma::uvec reorder_idx = reorder_row2col(2, 3);
        arma::uvec expected_idx = {0, 3, 1, 4, 2, 5};
        expect_true(arma::approx_equal(
            arma::conv_to<arma::vec>::from(reorder_idx),
            arma::conv_to<arma::vec>::from(expected_idx),
            "absdiff", tol));

        arma::mat A = {
            {1.0, 2.0},
            {3.0, 4.0}};
        arma::mat B(2, 1);
        B.col(0) = arma::vec({5.0, 6.0});
        arma::mat expected_diag_join = {
            {1.0, 2.0, 0.0},
            {3.0, 4.0, 0.0},
            {0.0, 0.0, 5.0},
            {0.0, 0.0, 6.0}};
        expect_true(compare_mat(diag_join(A, B), expected_diag_join, tol));

        arma::mat expected_grid = {
            {1.0, 4.0},
            {2.0, 4.0},
            {1.0, 5.0},
            {2.0, 5.0}};
        expect_true(compare_mat(
            expand_grid_2d(arma::vec({1.0, 2.0}), arma::vec({4.0, 5.0})),
            expected_grid, tol));
    }

    test_that("Linear algebra matrix transformations behave as expected")
    {
        const double tol = 1e-10;

        arma::mat jittered = {
            {1.0, 2.0},
            {2.0, 5.0}};
        jitter_mat(jittered, 0.25);
        arma::mat expected_jittered = {
            {1.25, 2.0},
            {2.0, 5.25}};
        expect_true(compare_mat(jittered, expected_jittered, tol));

        arma::mat asymmetric = {
            {1.0, 4.0},
            {2.0, 3.0}};
        make_symmetric(asymmetric);
        arma::mat expected_symmetric = {
            {1.0, 3.0},
            {3.0, 3.0}};
        expect_true(compare_mat(asymmetric, expected_symmetric, tol));

        arma::mat cov;
        arma::mat cov_inv;
        arma::mat diagonal = arma::diagmat(arma::vec({2.0, 4.0}));
        stabalized_inv(cov, cov_inv, diagonal);
        expect_true(compare_mat(cov, diagonal, tol));
        expect_true(compare_mat(
            cov_inv,
            arma::diagmat(arma::vec({0.5, 0.25})),
            tol));

        arma::mat spd = {
            {2.0, 0.5},
            {0.5, 1.5}};
        arma::mat spd_inv;
        fast_inv(spd_inv, spd);
        expect_true(compare_mat(spd_inv, arma::inv_sympd(spd), tol));

        arma::mat chol_cov = {
            {2.0, 0.0},
            {1.0, 3.0}};
        arma::mat reconstructed_cov;
        construct_cov(reconstructed_cov, chol_cov);
        arma::mat expected_cov = chol_cov * chol_cov.t();
        expect_true(compare_mat(reconstructed_cov, expected_cov, tol));
        expect_true(compare_mat(identity(3), arma::eye(3, 3), tol));

        expect_true(compare_double(
            log_det_sympd_cust(spd),
            log_det_chol(arma::chol(spd, "lower")),
            tol));

        arma::mat rhs = {
            {1.0, 3.0},
            {2.0, 4.0}};
        expect_true(compare_mat(
            chol_left_solve(chol_cov, rhs),
            arma::solve(arma::trimatl(chol_cov), rhs, arma::solve_opts::fast),
            tol));
    }

    test_that("Linear algebra Cholesky updates match direct decompositions")
    {
        const double tol = 1e-10;

        arma::mat base_cov = {
            {2.0, 0.5},
            {0.5, 1.5}};
        arma::mat base_chol = arma::chol(base_cov, "lower");

        arma::vec update_vec = {0.2, -0.4};
        arma::mat rank_one_chol = base_chol;
        chol_rank_one_update(rank_one_chol, 1.0, update_vec);
        expect_true(compare_mat(
            rank_one_chol * rank_one_chol.t(),
            base_cov + update_vec * update_vec.t(),
            tol));

        arma::mat update_mat = {
            {0.2, -0.1},
            {-0.4, 0.3}};
        arma::mat rank_n_chol = chol_rank_n_update(base_chol, 1.0, update_mat);
        arma::mat rank_n_ip_chol = base_chol;
        chol_rank_n_update_ip(rank_n_ip_chol, 1.0, update_mat);
        arma::mat expected_rank_n_cov = base_cov + update_mat * update_mat.t();
        expect_true(compare_mat(rank_n_chol * rank_n_chol.t(),
                                expected_rank_n_cov, tol));
        expect_true(compare_mat(rank_n_ip_chol * rank_n_ip_chol.t(),
                                expected_rank_n_cov, tol));

        arma::mat full_chol = {
            {2.0, 0.0, 0.0, 0.0},
            {0.5, 1.5, 0.0, 0.0},
            {-0.2, 0.3, 1.2, 0.0},
            {0.4, -0.1, 0.2, 1.1}};
        arma::mat full_cov = full_chol * full_chol.t();

        arma::mat expanded = full_chol.submat(0, 0, 1, 1);
        arma::mat lower_block = full_cov.submat(2, 0, 3, 1);
        arma::mat lower_diag = full_cov.submat(2, 2, 3, 3);

        add_cholesky_lower(expanded, lower_block, lower_diag);

        expect_true(compare_mat(expanded, full_chol, tol));
        expect_true(compare_mat(chol_of_sum(base_chol, arma::eye(2, 2)),
                                arma::chol(base_cov + arma::eye(2, 2), "lower"),
                                tol));
    }

    test_that("Woodbury inverse produces same result as general inverse")
    {
        set_r_seed(1);

        arma::mat A = arma::iwishrnd(identity(2), 5.0);
        arma::mat B = arma::iwishrnd(identity(5), 6.0);
        arma::mat U(2, 5, arma::fill::randn);

        arma::mat A_inv = arma::inv_sympd(A);
        arma::mat B_inv = arma::inv_sympd(B);

        arma::mat res1 = arma::inv_sympd(A + U * B * U.t());
        arma::mat res2 = woodbury_inv(A_inv, U, B_inv);

        expect_true(compare_mat(res1, res2, 1e-10));
    };
}
