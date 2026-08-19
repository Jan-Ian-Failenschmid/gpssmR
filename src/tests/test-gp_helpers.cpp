// [[Rcpp::depends(RcppArmadillo)]]

#include <testthat.h>
#include <RcppArmadillo.h>
#include "test_helper.h"
#include "linear_algebra.h"
#include "hsgp_struct.h"
#include "kernel_helper.h"

context("C++ GP helpers")
{
    // Eigenfunctions and values on a rectangular domain with dirichlet boundary
    test_that("HSGP square-root eigenvalues are calculated correctly")
    {
        const double tol = 1e-10;

        const arma::rowvec L = {
            2.0,
            3.0
        };

        const arma::mat m = {
            {1.0, 1.0},
            {2.0, 1.0}
        };

        const arma::mat expected_sqrt_lambda = {
            {M_PI / 4.0, M_PI / 6.0},
            {M_PI / 2.0, M_PI / 6.0}
        };

        expect_true(compare_mat(
                gp_sqrt_lambda_nd_vec(L, m), expected_sqrt_lambda, tol));
    }

    test_that("HSGP basis functions are calculated correctly")
    {
        const double tol = 1e-10;

        const arma::vec boundary_factor = {2.0, 3.0};
        const arma::vec sqrt_lambda = {M_PI / 4.0, M_PI / 6.0};

        const arma::mat X = {
            {0.0, 0.5},
            {-1.0, 1.0}
        };

        const arma::vec inv_sqrt_L = arma::sqrt(1.0 / boundary_factor);

        const arma::mat sin_X =
            arma::sin(
                sqrt_lambda % (X.each_col() + boundary_factor).each_col()
            );

        const arma::rowvec expected_phi = 
            arma::prod(sin_X.each_col() % inv_sqrt_L, 0);

        expect_true(
            compare_mat(
                gp_phi_nD(boundary_factor, sqrt_lambda, X),
                expected_phi, tol));
    }

    // Squared exponential
    test_that("GP SE covariance helpers match direct kernel calculations")
    {
        const double tol = 1e-10;

        arma::mat x1 = {
            {0.0, 1.0},
            {1.0, 2.0}
        };

        arma::mat x2 = {
            {0.0, 2.0},
            {1.0, 3.0}
        };

        const double rho = 1.5;
        const double alpha = 2.0;

        squared_exponential kernel;

        arma::mat expected_cross(x1.n_cols, x2.n_cols);

        for (arma::uword i = 0; i < x1.n_cols; ++i)
        {
            for (arma::uword j = 0; j < x2.n_cols; ++j)
            {
                const arma::vec diff = x1.col(i) - x2.col(j);

                expected_cross(i, j) =
                    alpha * alpha *
                    std::exp(-0.5 * arma::dot(diff, diff) / (rho * rho));
            }
        }

        expect_true(compare_mat(
                kernel.gp_covariance_multi(x1, x2, arma::vec({ alpha, rho })),
                expected_cross, tol));

        arma::mat expected_self(x1.n_cols, x1.n_cols);

        for (arma::uword i = 0; i < x1.n_cols; ++i)
        {
            for (arma::uword j = 0; j < x1.n_cols; ++j)
            {
                const arma::vec diff = x1.col(i) - x1.col(j);

                expected_self(i, j) =
                    alpha * alpha *
                    std::exp(-0.5 * arma::dot(diff, diff) / (rho * rho));
            }
        }

        expect_true(compare_mat(
                kernel.gp_covariance_multi(x1, arma::vec({ alpha, rho })),
                expected_self, tol));
    }

    test_that("HSGP SE spectral density matches analytic calculation")
    {
        const double tol = 1e-10;

        const arma::mat Lambda = {
            {M_PI / 4.0, M_PI / 6.0},
            {M_PI / 2.0, M_PI / 6.0}
        };

        const double alpha = 1.5;
        const double rho = 0.75;

        const double dim = static_cast<double>(Lambda.n_cols);

        const arma::vec squared_norms = arma::sum(arma::square(Lambda), 1);

        const arma::vec expected_spdf = alpha * alpha *
            std::pow(std::sqrt(2.0 * M_PI) * rho, dim) *
            arma::exp(-0.5 * rho * rho * squared_norms);

        squared_exponential kernel;

        expect_true(
            compare_mat(
                kernel.gp_spdf_nd_vec(Lambda, arma::vec({ alpha, rho })),
                expected_spdf, tol));
    }

    // HSGP structure
    test_that("HSGP is structually consistent")
    {
        const double tol = 1e-10;

        const arma::mat m = {
            {1.0, 1.0},
            {2.0, 1.0}
        };

        const arma::vec boundary_factor = {2.0, 3.0};

        const arma::mat X = {
            {0.0, 0.5},
            {-1.0, 1.0}
        };

        const double alpha = 1.5;
        const double rho = 0.75;

        squared_exponential kernel;

        const arma::vec expected_spdf = kernel.gp_spdf_nd_vec(
                gp_sqrt_lambda_nd_vec(boundary_factor.t(), m),
                arma::vec({ alpha, rho }));

        hsgp_approx hsgp(m, boundary_factor,
            std::make_unique<squared_exponential>());

        hsgp.set_hyperparameters(alpha, rho);

        const arma::mat basis = hsgp.basis_functions(X);
        const arma::mat scaled_basis = hsgp.scaled_basis_functions(X);

        expect_true(compare_mat(*hsgp.get_predictor_ptr(), basis, tol));

        expect_true(compare_mat(
                hsgp.scale() * hsgp.inv_scale(),
            identity(hsgp.scale().n_rows), tol));

        expect_true(compare_mat(
                (*hsgp.get_cov_chol_ptr()) * hsgp.get_cov_chol_ptr()->t(),
            hsgp.scale(), tol));

        expect_true(compare_mat(
                scaled_basis,arma::diagmat(expected_spdf) * basis, tol));
    }

    // Matern kernel
    test_that("Matern covariance has alpha squared marginal variance")
    {
        const double tol = 1e-12;

        const arma::vec x = {0.5, -1.2, 2.0};

        const double alpha = 2.0;
        const double rho = 1.5;
        const double alpha_sq = alpha * alpha;
        const arma::vec hyperparameters = {alpha, rho};

        matern matern12(arma::vec({ 0.5 }));
        matern matern32(arma::vec({ 1.5 }));
        matern matern52(arma::vec({ 2.5 }));

        expect_true(compare_double(
            matern12.covariance(x, x, hyperparameters),
            alpha_sq, tol));

        expect_true(compare_double(
            matern32.covariance(x, x, hyperparameters),
            alpha_sq, tol));

        expect_true(compare_double(
                matern52.covariance(x, x, hyperparameters), alpha_sq, tol));
    }


    test_that("General Matern matches known analytic covariances")
    {
        const double tol = 1e-10;

        const arma::vec x1 = {0.0, 1.0};

        const arma::vec x2 = {1.0, 2.5};

        const double alpha = 2.0;
        const double alpha_sq = alpha * alpha;
        const double rho = 1.5;
        const arma::vec hyperparameters = arma::vec({ alpha, rho });

        const double distance = arma::norm(x1 - x2, 2);

        // 1/2
        matern matern12(arma::vec({ 0.5 }));
        double expected = alpha_sq * std::exp(-distance / rho);
        double actual = matern12.covariance(x1, x2, hyperparameters);
        expect_true(compare_double(actual, expected, tol));

        // 3/2
        matern matern32(arma::vec({ 1.5 }));
        double z = std::sqrt(3.0) * distance / rho;
        expected = alpha_sq * (1.0 + z) * std::exp(-z);
        actual = matern32.covariance(x1, x2, hyperparameters);
        expect_true(compare_double(actual, expected, tol));

        // 5/2
        matern matern52(arma::vec({ 2.5 }));
        z = std::sqrt(5.0) * distance / rho;
        expected = alpha_sq * (1.0 + z + z * z / 3.0) * std::exp(-z);
        actual = matern52.covariance(x1, x2, hyperparameters);
        expect_true(compare_double(actual, expected, tol));
    }


    test_that("Matern covariance helpers match analytic Matern 3/2 matrix")
    {
        const double tol = 1e-10;

        const arma::mat x1 = {
            {0.0, 1.0},
            {1.0, 2.0}
        };

        const arma::mat x2 = {
            {0.0, 2.0},
            {1.0, 3.0}
        };

        const double alpha = 2.0;
        const double alpha_sq = alpha * alpha;
        const double rho = 1.5;
        const arma::vec hyperparameters = arma::vec({ alpha, rho });

        matern kernel(arma::vec({ 1.5 }));

        double distance;
        double z;

        arma::mat expected_cross(x1.n_cols, x2.n_cols);

        for (arma::uword i = 0; i < x1.n_cols; ++i)
        {
            for (arma::uword j = 0; j < x2.n_cols; ++j)
            {
                distance = arma::norm(x1.col(i) - x2.col(j), 2);
                z = std::sqrt(3.0) * distance / rho;
                expected_cross(i, j) = alpha_sq * (1.0 + z) * std::exp(-z);
            }
        }

        expect_true(compare_mat(
                kernel.gp_covariance_multi(x1, x2, hyperparameters),
                expected_cross, tol));

        arma::mat expected_self(x1.n_cols, x1.n_cols);

        for (arma::uword i = 0; i < x1.n_cols; ++i)
        {
            for (arma::uword j = 0; j < x1.n_cols; ++j)
            {
                distance = arma::norm(x1.col(i) - x1.col(j), 2);
                z = std::sqrt(3.0) * distance / rho;
                expected_self(i, j) = alpha_sq * (1.0 + z) * std::exp(-z);
            }
        }

        expect_true(compare_mat(
            kernel.gp_covariance_multi(x1, hyperparameters),
                expected_self, tol));
    }


    test_that("HSGP Matern 3/2 spectral density matches analytic calculation")
    {
        const double tol = 1e-10;

        const arma::mat Lambda = {
            {M_PI / 4.0, M_PI / 6.0},
            {M_PI / 2.0, M_PI / 6.0},
            {3.0 * M_PI / 4.0, M_PI / 3.0}
        };

        const double alpha = 1.5;
        const double rho = 0.75;
        const double nu = 1.5;

        const double dim = static_cast<double>(Lambda.n_cols);

        const arma::vec squared_norms = arma::sum(arma::square(Lambda), 1);
        
        // Commented out sections align with the calculations in the kernel 
        // and the calculations in the test are the analyitical solutons for 
        // nu = 3/2. All sections are interchangable. 
        // const double exponent = nu + dim / 2.0;
        const double exponent = (3 + dim) / 2.0;

        // const double numerator = std::pow(2.0, dim) *
        //     std::pow(M_PI, dim / 2.0) * std::tgamma(exponent) *
        //     std::pow(2.0 * nu, nu);
        const double numerator = std::pow(2.0, dim) *
            std::pow(M_PI, dim / 2.0) * std::tgamma(exponent) *
            std::pow(3, nu);

        // const double denominator = std::tgamma(nu) * std::pow(rho, 2.0 * nu);
        const double denominator = 0.5 * std::sqrt(M_PI) * std::pow(rho, 3);

        const double scale = alpha * alpha * numerator / denominator;

        // const arma::vec expected =
        //     scale * arma::pow(2.0 * nu / (rho * rho) + squared_norms,
        //         -exponent);
        const arma::vec expected =
            scale * arma::pow(3 / (rho * rho) + squared_norms,
                -exponent);

        matern kernel(arma::vec({ nu }));

        expect_true(compare_mat(
                kernel.gp_spdf_nd_vec(Lambda, arma::vec({ alpha, rho })),
                expected, tol));
    }
}
