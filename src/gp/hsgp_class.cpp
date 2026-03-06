// [[Rcpp::depends(RcppArmadillo)]]

#include <RcppArmadillo.h>
#include "hsgp_class.h"
#include "hsgp_helper.h"

// Constructor
hsgp_approx::hsgp_approx(
    arma::mat indices_inp,       // Matrix of indices in each dim
    arma::vec boundry_factor_inp // Boundry factor in each dim
    ) : indices(indices_inp), boundry_factor(boundry_factor_inp), alpha(),
        rho(), spdf(), phi()
{
    sqrt_lambda = gp_sqrt_lambda_nd_vec(boundry_factor.t(), indices);
}

void hsgp_approx::update_hyperparameters(
    const double &alpha_new, const double &rho_new)
{
    alpha = alpha_new;
    rho = rho_new;
    spdf = gp_spdf_nd_vec(sqrt_lambda, alpha, rho);
}

void hsgp_approx::phi_transform(const arma::mat &x)
{
    uint n_phi = sqrt_lambda.n_rows;
    phi.set_size(n_phi, x.n_cols);

    for (size_t i = 0; i < n_phi; i++)
    {
        phi.row(i) = gp_phi_nD(boundry_factor, sqrt_lambda.row(i).t(), x);
    }
}

arma::mat hsgp_approx::scaled_basis_functions(const arma::mat &x)
{
    phi_transform(x);
    return arma::diagmat(spdf) * phi;
}

arma::mat hsgp_approx::basis_functions(const arma::mat &x)
{
    phi_transform(x);
    return phi;
}

arma::mat hsgp_approx::inv_scale()
{
    // Calculate inverse spdf
    arma::vec inv_spdf = 1.0 / spdf;

    // Replace inf with machine maximim for stability if spdf is
    // numerically 0
    for (size_t i = 0; i < inv_spdf.n_elem; i++)
    {
        if (std::isinf(inv_spdf(i)))
        {
            inv_spdf(i) = std::numeric_limits<double>::max();
        }
    }

    return arma::diagmat(inv_spdf);
}

arma::mat hsgp_approx::scale()
{
    // Calculate inverse spdf
    arma::vec inv_spdf = 1.0 / spdf;

    // Adjustment for numerical stability
    for (size_t i = 0; i < inv_spdf.n_elem; i++)
    {
        if (std::isinf(inv_spdf(i)))
        {
            inv_spdf(i) = std::numeric_limits<double>::max();
        }
    }

    // Rvert to spdf and return
    return arma::diagmat(1.0 / inv_spdf);
}