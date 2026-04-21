// [[Rcpp::depends(RcppArmadillo)]]

#include <testthat.h>
#include <RcppArmadillo.h>
#include "test_helper.h"
#include "linear_algebra.h"
#include "hsgp_helper.h"
#include "hsgp_struct.h"
#include "imc_gp_helper.h"

context("C++ GP helpers")
{
    test_that("GP covariance helpers match direct kernel calculations")
    {
        const double tol = 1e-10;

        arma::mat x1 = {
            {0.0, 1.0},
            {1.0, 2.0}};
        arma::mat x2 = {
            {0.0, 2.0},
            {1.0, 3.0}};
        double rho = 1.5;
        double alpha = 2.0;

        arma::mat expected_cross(2, 2);
        for (arma::uword i = 0; i < x1.n_cols; i++)
        {
            for (arma::uword j = 0; j < x2.n_cols; j++)
            {
                arma::vec diff = x1.col(i) - x2.col(j);
                expected_cross(i, j) =
                    std::pow(alpha, 2) *
                    std::exp(-0.5 * arma::dot(diff, diff) / std::pow(rho, 2));
            }
        }

        expect_true(compare_mat(
            gp_covariance_multi(x1, x2, rho, alpha), expected_cross, tol));

        arma::mat expected_self(2, 2);
        for (arma::uword i = 0; i < x1.n_cols; i++)
        {
            for (arma::uword j = 0; j < x1.n_cols; j++)
            {
                arma::vec diff = x1.col(i) - x1.col(j);
                expected_self(i, j) =
                    std::pow(alpha, 2) *
                    std::exp(-0.5 * arma::dot(diff, diff) / std::pow(rho, 2));
            }
        }
        expect_true(compare_mat(
            gp_covariance_multi(x1, rho, alpha), expected_self, tol));
    }

    test_that("HSGP helper functions and struct are internally consistent")
    {
        const double tol = 1e-10;

        arma::rowvec L = {2.0, 3.0};
        arma::mat m = {
            {1.0, 1.0},
            {2.0, 1.0}};
        arma::mat expected_sqrt_lambda = {
            {M_PI / 4.0, M_PI / 6.0},
            {M_PI / 2.0, M_PI / 6.0}};
        expect_true(compare_mat(
            gp_sqrt_lambda_nd_vec(L, m), expected_sqrt_lambda, tol));

        double alpha = 1.5;
        double rho = 0.75;
        arma::vec expected_spdf =
            std::pow(alpha, 2) *
            std::pow(std::sqrt(2.0 * M_PI) * rho,
                     static_cast<double>(expected_sqrt_lambda.n_cols)) *
            arma::exp(-0.5 * std::pow(rho, 2) *
                      arma::sum(arma::square(expected_sqrt_lambda), 1));
        expect_true(compare_mat(
            gp_spdf_nd_vec(expected_sqrt_lambda, alpha, rho),
            expected_spdf,
            tol));

        arma::vec boundary_factor = {2.0, 3.0};
        arma::vec sqrt_lambda = expected_sqrt_lambda.row(0).t();
        arma::mat X = {
            {0.0, 0.5},
            {-1.0, 1.0}};
        arma::vec inv_sqrt_L = arma::sqrt(1.0 / boundary_factor);
        arma::mat sin_X = arma::sin(
            sqrt_lambda % (X.each_col() + boundary_factor).each_col());
        arma::rowvec expected_phi = arma::prod(sin_X.each_col() % inv_sqrt_L, 0);
        expect_true(compare_mat(gp_phi_nD(boundary_factor, sqrt_lambda, X),
                                expected_phi, tol));

        hsgp_approx hsgp(m, boundary_factor);
        hsgp.set_hyperparameters(alpha, rho);
        arma::mat basis = hsgp.basis_functions(X);
        arma::mat scaled_basis = hsgp.scaled_basis_functions(X);

        expect_true(compare_mat(*hsgp.get_predictor_ptr(),
                                basis,
                                tol));
        expect_true(compare_mat(
            hsgp.scale() * hsgp.inv_scale(),
            identity(hsgp.scale().n_rows),
            1e-8));
        expect_true(compare_mat(
            (*hsgp.get_cov_chol_ptr()) * hsgp.get_cov_chol_ptr()->t(),
            hsgp.scale(),
            1e-8));
        expect_true(compare_mat(
            scaled_basis,
            arma::diagmat(expected_spdf) * basis,
            1e-8));
    }
}
