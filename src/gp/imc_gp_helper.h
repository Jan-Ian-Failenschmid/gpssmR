#ifndef IMC_GP_HELPER_H
#define IMC_GP_HELPER_H

// [[Rcpp::depends(RcppArmadillo)]]

// Squared Exponential Kernel
inline arma::mat gp_covariance_multi(const arma::mat &x1, const arma::mat &x2,
                              double length_scale,
                              double signal_variance)
{
    const size_t n1 = x1.n_cols;
    const size_t n2 = x2.n_cols;
    arma::mat K(n1, n2);
    arma::vec diff(x1.n_rows);
    double sqdist;

    const double inv_ell2 = 1 / std::pow(length_scale, 2);
    const double sig2 = std::pow(signal_variance, 2);

    for (size_t i = 0; i < n1; ++i)
    {
        for (size_t j = 0; j < n2; ++j)
        {
            diff = x1.col(i) - x2.col(j);
            sqdist = arma::dot(diff, diff);
            K(i, j) = sig2 * std::exp(-0.5 * sqdist * inv_ell2);
        }
    }
    return K;
}

inline arma::mat gp_covariance_multi(const arma::mat &x,
                                     double length_scale,
                                     double signal_variance)
{
    const size_t n = x.n_cols;
    arma::mat K(n, n);
    arma::vec diff(x.n_rows);
    double sqdist;

    const double inv_ell2 = 1.0 / std::pow(length_scale, 2);
    const double sig2 = std::pow(signal_variance, 2);

    for (size_t i = 0; i < n; ++i)
    {
        K(i, i) = sig2;

        for (size_t j = 0; j < i; ++j)
        {
            diff = x.col(i) - x.col(j);
            sqdist = arma::dot(diff, diff);

            K(i, j) = sig2 * std::exp(-0.5 * sqdist * inv_ell2);
            K(j, i) = K(i, j); // mirror
        }
    }

    return K;
}

#endif