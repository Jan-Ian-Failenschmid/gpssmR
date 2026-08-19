// [[Rcpp::depends(RcppArmadillo)]]

#include "kernel_helper.h"
#include <Rmath.h>

// Kernel base
arma::mat kernel_base::gp_covariance_multi(
    const arma::mat& x,
    const arma::vec& hyperparameters) const
{
    const arma::uword n = x.n_cols;
    arma::mat K(n, n);

    for (arma::uword i = 0; i < n; ++i)
    {
        K(i, i) = covariance(x.col(i), x.col(i), hyperparameters);

        for (arma::uword j = 0; j < i; ++j)
        {
            K(i, j) = covariance(x.col(i), x.col(j), hyperparameters);
            K(j, i) = K(i, j); // mirror
        }
    }

    return K;
}

arma::mat kernel_base::gp_covariance_multi(
    const arma::mat& x1, 
    const arma::mat& x2,
    const arma::vec& hyperparameters) const
{
    const arma::uword n1 = x1.n_cols;
    const arma::uword n2 = x2.n_cols;
    
    arma::mat K(n1, n2);

    for (arma::uword i = 0; i < n1; ++i)
    {
        for (arma::uword j = 0; j < n2; ++j)
        {
            K(i, j) = covariance(x1.col(i), x2.col(j), hyperparameters);
        }
    }
    return K;
}

// Squared exponential kernel
arma::vec squared_exponential::gp_spdf_nd_vec(
    const arma::mat& Lambda, 
    const arma::vec& hyperparameters) const
{
    const double alpha = hyperparameters[0];
    const double rho = hyperparameters[1];

    // ||lambda||^2 for each row
    arma::vec norms = arma::sum(arma::square(Lambda), 1);
    const double dim = static_cast<double>(Lambda.n_cols);
    const double scale =
        std::pow(alpha, 2) *
        std::pow(std::sqrt(2.0 * M_PI) * rho, dim);
    arma::vec result = scale * arma::exp(-0.5 * rho * rho * norms);
    return result;
}

double squared_exponential::covariance(
    const arma::vec& x1, 
    const arma::vec& x2, 
    const arma::vec& hyperparameters) const
{
    const double alpha = hyperparameters[0];
    const double rho = hyperparameters[1];

    const double sig2 = std::pow(alpha, 2);
    const double inv_ell2 = 1 / std::pow(rho, 2);

    // Using norm + sqrt results in negligable numeric differences.
    arma::vec diff = x1 - x2;
    double sq_dist = arma::dot(diff, diff);
    
    return sig2 * std::exp(-0.5 * sq_dist * inv_ell2);
}

// Matern kernel
arma::vec matern::gp_spdf_nd_vec(
    const arma::mat& Lambda,
    const arma::vec& hyperparameters) const
{
    const double alpha = hyperparameters[0];
    const double rho = hyperparameters[1];
    const double nu = fixed_pars[0];

    // ||lambda||^2 for each row
    arma::vec norms = arma::sum(arma::square(Lambda), 1);
    const double dim = static_cast<double>(Lambda.n_cols);

    const double exponent = nu + dim / 2.0;

    const double numerator = std::pow(2.0, dim) * std::pow(M_PI, dim / 2.0) *
        std::tgamma(exponent) * std::pow(2.0 * nu, nu);

    const double denominator = std::tgamma(nu) * std::pow(rho, 2.0 * nu);

    const double scale = alpha * alpha * numerator / denominator;

    return scale * arma::pow(2.0 * nu / (rho * rho) + norms, -exponent);
}

double matern::covariance(
    const arma::vec& x1,
    const arma::vec& x2,
    const arma::vec& hyperparameters) const
{
    const double alpha = hyperparameters[0];
    const double rho = hyperparameters[1];
    const double nu = fixed_pars[0];

    const double sig2 = std::pow(alpha, 2);

    const arma::vec diff = x1 - x2;
    double eucl_dist = arma::norm(diff, 2);

    if (eucl_dist == 0.0)
    {
        return sig2;
    }

    const double z = std::sqrt(2.0 * nu) * eucl_dist / rho;

    const double normalizing_constant =
        std::pow(2.0, 1.0 - nu) / std::tgamma(nu);

    const double bessel_value = R::bessel_k(z, nu, 1.0);

    return sig2 * normalizing_constant * std::pow(z, nu) * bessel_value;
}


// Kernel factory 
std::unique_ptr<kernel_base> make_kernel(
    const Rcpp::List& kernel_spec) 
{
    const std::string kernel_name = Rcpp::as<std::string>(kernel_spec["name"]);
    const arma::vec constants = Rcpp::as<arma::vec>(
        kernel_spec["constants"]);;


    if (kernel_name == "squared_exponential")
    {
        return std::make_unique<squared_exponential>(constants);
    }
    if (
        kernel_name == "matern12" ||
        kernel_name == "matern32" ||
        kernel_name == "matern52")
    {
        return std::make_unique<matern>(constants);;
    }

    return nullptr;
}