#ifndef KERNEL_HELPER_H
#define KERNEL_HELPER_H

#include <RcppArmadillo.h>

// HSGP eigenfunctions and values 
// Rectangular domain with Dirichlet Boundary

inline arma::mat gp_sqrt_lambda_nd_vec(
    const arma::rowvec& L, // Boundry factor in each dimension
    const arma::mat& m     // Basis function indicator in each dimension
)
{
    arma::mat m_pi = m * M_PI;
    // return arma::square(m_pi.each_row() / (2 * L));
    return m_pi.each_row() / (2 * L); // Return square root of lambda directly
}

// PHI basis function transformation -----
// Calculate jth PHI basis function transform of State Variables over time
// or particles
inline arma::rowvec gp_phi_nD(
    const arma::vec& L,
    const arma::vec& sqrt_lambda,
    const arma::mat& X_mat)
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

// Kernel base struct
struct kernel_base
{
    arma::vec fixed_pars;
    
    kernel_base(const arma::vec& fixed_pars_ = arma::vec()) : 
        fixed_pars(fixed_pars_) {};

    virtual ~kernel_base() = default; 

    arma::mat gp_covariance_multi(
        const arma::mat& x,
        const arma::vec& hyperparameters) const;
    arma::mat gp_covariance_multi(
        const arma::mat& x1, const arma::mat& x2,
        const arma::vec& hyperparameters) const;
    virtual double covariance(
        const arma::vec& x1,
        const arma::vec& x2,
        const arma::vec& hyperparameters) const = 0;
    virtual arma::vec gp_spdf_nd_vec(
        const arma::mat& Lambda, 
        const arma::vec& hyperparameters) const = 0;

    void set_fixed_pars(const arma::vec& fixed_pars_) {
        fixed_pars = fixed_pars_;
    };
};

struct squared_exponential : public kernel_base
{
    using kernel_base::kernel_base;

    double covariance(
        const arma::vec& x1,
        const arma::vec& x2,
        const arma::vec& hyperparameters) const override;
    arma::vec gp_spdf_nd_vec(
        const arma::mat& Lambda, 
        const arma::vec& hyperparameters) const override;
};

struct matern : public kernel_base
{
    using kernel_base::kernel_base;

    double covariance(
        const arma::vec& x1,
        const arma::vec& x2,
        const arma::vec& hyperparameters) const override;
    arma::vec gp_spdf_nd_vec(
        const arma::mat& Lambda,
        const arma::vec& hyperparameters) const override;
};


// Kernel factory
std::unique_ptr<kernel_base> make_kernel(
    const Rcpp::List& kernel_spec);

#endif