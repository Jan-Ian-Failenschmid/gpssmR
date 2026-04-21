#ifndef HSGP_HELPER_H
#define HSGP_HELPER_H

#include <RcppArmadillo.h>

inline arma::mat gp_sqrt_lambda_nd_vec(
    const arma::rowvec &L, // Boundry factor in each dimension
    const arma::mat &m     // Basis function indicator in each dimension
)
{
    arma::mat m_pi = m * M_PI;
    // return arma::square(m_pi.each_row() / (2 * L));
    return m_pi.each_row() / (2 * L); // Return square root of lambda directly
}

// Calculate spectral density kernel ---wd--
// Calculate kernel of the spectral density for HSGP approximation
inline arma::vec gp_spdf_nd_vec(
    const arma::mat &Lambda, // Matrix where each row is a lambda vector
    const double &alpha,     // Marginal variance
    const double &rho        // Length scale
)
{
    // ||lambda||^2 for each row
    arma::vec norms = arma::sum(arma::square(Lambda), 1);
    const double dim = static_cast<double>(Lambda.n_cols);
    const double scale =
        std::pow(alpha, 2) *
        std::pow(std::sqrt(2.0 * M_PI) * rho, dim);
    arma::vec result = scale * arma::exp(-0.5 * rho * rho * norms);
    return result;
}

// PHI basis function transformation -----
// Calculate jth PHI basis function transform of State Variables over time
// or particles
inline arma::rowvec gp_phi_nD(
    const arma::vec &L,
    const arma::vec &sqrt_lambda,
    const arma::mat &X_mat)
{
    // Calculate PHI j over all columns of a D dimensional state matrix (n, T)
    // Riutort-Mayol et al., 2023 EQ: 12
    // Compute sqrt(lambda) and 1/sqrt(L) once
    arma::vec inv_sqrt_L = arma::sqrt(1.0 / L);

    // Add L to each column of X and multiply each column
    // element-wise by sqrt_lambda
    arma::mat sin_X = arma::sin(
        sqrt_lambda % (X_mat.each_col() + L).each_col());

    // Scale by 1/sqrt(L)
    arma::mat phi_j = sin_X.each_col() % inv_sqrt_L;

    // Compute the product along each column
    return arma::prod(phi_j, 0);
}

#endif
