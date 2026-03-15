// [[Rcpp::depends(RcppArmadillo)]]

#include <RcppArmadillo.h>
#include "hsgp_struct.h"
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
    spdf = gp_spdf_nd_vec(sqrt_lambda, alpha, rho) + 1e-300;
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

arma::mat *hsgp_approx::get_predictor_ptr()
{
    return &phi;
}

arma::mat *hsgp_approx::get_cov_chol_ptr()
{
    return &scale_chol;
}

void hsgp_approx::set_chol()
{
    scale_chol = arma::diagmat(arma::sqrt(spdf));
}

arma::mat hsgp_approx::inv_scale()
{
    return arma::diagmat(1.0 / spdf);
}

arma::mat hsgp_approx::scale()
{
    // Rvert to spdf and return
    return arma::diagmat(spdf);
}