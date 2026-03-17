#ifndef PROBABILITY_DISTRIBUTIONS_H
#define PROBABILITY_DISTRIBUTIONS_H

#include <RcppArmadillo.h>
#include "linear_algebra.h"

using namespace Rcpp;

inline double logdnorm(const arma::vec &x, const arma::vec &mu,
                       const arma::mat &chol_cov)
{
    arma::vec diff = x - mu;
    arma::vec z = chol_left_solve(chol_cov, diff);

    double quad = arma::dot(z, z);
    double normconst = -0.5 * chol_cov.n_cols * log(2 * M_PI) -
                       0.5 * log_det_chol(chol_cov);

    return normconst - 0.5 * quad;
}

inline arma::vec mat_logdnorm_inv_kernel(const arma::mat &diff,
                                         const arma::mat &inv_cov,
                                         double log_normconst)
{
    return log_normconst - 0.5 * arma::trans(
                                     arma::sum((inv_cov * diff) % diff, 0));
}

inline arma::vec mat_logdnorm_unnorm_inv(const arma::vec &x,
                                         const arma::mat &mu,
                                         const arma::mat &inv_cov)
{
    arma::mat diff = x - mu.each_col();
    return mat_logdnorm_inv_kernel(diff, inv_cov, 0.0);
}

inline arma::vec mat_logdnorm_unnorm_inv(const arma::mat &x,
                                         const arma::vec &mu,
                                         const arma::mat &inv_cov)
{
    arma::mat diff = x.each_col() - mu;
    return mat_logdnorm_inv_kernel(diff, inv_cov, 0.0);
}

inline arma::vec mat_logdnorm_unnorm_inv(const arma::mat &x,
                                         const arma::mat &mu,
                                         const arma::mat &inv_cov)
{
    arma::mat diff = x - mu;
    return mat_logdnorm_inv_kernel(diff, inv_cov, 0.0);
}

inline arma::vec mat_logdnorm_kernel(const arma::mat &diff, const arma::mat &chol_cov)
{
    arma::mat z = chol_left_solve(chol_cov, diff);

    arma::rowvec quad =
        arma::sum(z % z, 0);

    double normconst = -0.5 * chol_cov.n_cols * log(2 * M_PI) -
                       0.5 * log_det_chol(chol_cov);

    return normconst - 0.5 * arma::trans(quad);
}

inline arma::vec mat_logdnorm(const arma::vec &x,
                              const arma::mat &mu,
                              const arma::mat &chol_cov)
{
    arma::mat diff = x - mu.each_col();
    return mat_logdnorm_kernel(diff, chol_cov);
}

inline arma::vec mat_logdnorm(const arma::mat &x,
                              const arma::vec &mu,
                              const arma::mat &chol_cov)
{
    arma::mat diff = x.each_col() - mu;
    return mat_logdnorm_kernel(diff, chol_cov);
}

inline arma::vec mat_logdnorm(const arma::mat &x,
                              const arma::mat &mu,
                              const arma::mat &chol_cov)
{
    arma::mat diff = x - mu;
    return mat_logdnorm_kernel(diff, chol_cov);
}

inline void conditional_mvnormal(
    arma::vec &cond_mu,
    arma::vec &cond_sigma,
    const arma::vec &mu,
    const arma::mat &sigma,
    const arma::vec &fixed_vals,
    const arma::uvec &fixed, // Indicator for fixed element
    const arma::uvec &free   // Indicator for free elements
)
{
    arma::vec mu_free = mu(free);   // mu1
    arma::vec mu_fixed = mu(fixed); // mu2

    arma::mat sigma_free = sigma(free, free);        // Sigma11
    arma::mat sigma_free_fixed = sigma(free, fixed); // Sigma12
    arma::mat sigma_fixed_free = sigma(fixed, free); // Sigma21
    arma::mat sigma_fixed = sigma(fixed, fixed);     // Sigma22

    arma::mat inv_sigma_fixed = arma::pinv(sigma_fixed);
    arma::mat sff_inv_sf = sigma_free_fixed * inv_sigma_fixed;

    cond_mu = mu_free + sff_inv_sf * (fixed_vals - mu_fixed);
    cond_sigma = sigma_free - sff_inv_sf * sigma_fixed_free;
}

inline void mat_rnorm(
    arma::mat &X, const arma::mat &mu, const arma::mat &cov_chol)
{
    X.randn();
    X = cov_chol * X;
    X += mu;
};

inline arma::mat mat_rnorm(const arma::mat &mu, const arma::mat &cov_chol)
{
    arma::mat X(mu.n_rows, mu.n_cols);
    mat_rnorm(X, mu, cov_chol);
    return X;
};

// Matrix normal distribution
inline double logdmatnorm(
    const arma::mat &X,
    const arma::mat &mu,
    const arma::mat &row_chol, // lower chol of row_cov
    const arma::mat &col_chol) // lower chol of col_cov
{
    arma::mat diff = X - mu;

    // Solve row_chol * Z = diff
    arma::mat Z = chol_left_solve(row_chol, diff);

    // Solve col_chol * W = Z.t()
    arma::mat W = chol_left_solve(col_chol, Z.t());

    double quad = -0.5 * arma::accu(W % W); // Frobenius norm squared

    int D = row_chol.n_rows;
    int n = col_chol.n_rows;

    double logdet_row = log_det_chol(row_chol);
    double logdet_col = log_det_chol(col_chol);

    double log_norm =
        D * n * std::log(2.0 * M_PI) + n * logdet_row + D * logdet_col;

    return quad - 0.5 * log_norm;
}

inline void rmatnorm(
    arma::mat &X,
    const arma::mat &mu,
    const arma::mat &row_chol, // lower chol of row_cov
    const arma::mat &col_chol)
{
    X.randn();
    X = row_chol * X * col_chol.t();
    X += mu;
}

inline arma::mat rmatnorm(
    const arma::mat &mu,
    const arma::mat &row_chol, // lower chol of row_cov
    const arma::mat &col_chol)
{
    arma::mat X(mu.n_rows, mu.n_cols);
    rmatnorm(X, mu, row_chol, col_chol);
    return X;
}

// Wishart and Inverse wishart
// Multivariate log-gamma
inline double log_mvgamma(double a, int p)
{
    double out = p * (p - 1) * 0.25 * std::log(M_PI);
    for (int j = 1; j <= p; ++j)
    {
        out += std::lgamma(a + (1 - j) * 0.5);
    }
    return out;
}

// Inverse Wishart
inline double log_dmatrixt(
    const arma::mat &sigma_y_chol,
    const arma::mat &prior_cov_scale_chol,
    const arma::mat &post_cov_scale_chol,
    uint v_prior,
    uint v_post)
{
    uint dy = prior_cov_scale_chol.n_cols;
    uint n = v_post - v_prior;

    double log_det_y = log_det_chol(sigma_y_chol);
    double log_det_prior_cov_scale = log_det_chol(prior_cov_scale_chol);
    double log_det_post_cov_scale = log_det_chol(post_cov_scale_chol);

    double log_marg_likelihood =
        -std::log(M_PI) * (dy * n / 2.0) -
        (dy / 2.0) * log_det_y +
        (v_prior / 2.0) * log_det_prior_cov_scale -
        (v_post / 2.0) * log_det_post_cov_scale +
        log_mvgamma((v_post / 2.0), dy) -
        log_mvgamma((v_prior / 2.0), dy);

    return log_marg_likelihood;
}

inline double log_dmatrixt(
    const arma::mat &prior_col_chol,
    const arma::mat &posterior_col_chol,
    const arma::mat &prior_cov_scale_chol,
    const arma::mat &post_cov_scale_chol,
    uint v_prior,
    uint v_post)
{
    uint dy = prior_cov_scale_chol.n_cols;
    uint n = v_post - v_prior;

    double log_det_prior_col = log_det_chol(prior_col_chol);
    double log_det_posterior_col = log_det_chol(posterior_col_chol);
    double log_det_prior_cov_scale = log_det_chol(prior_cov_scale_chol);
    double log_det_post_cov_scale = log_det_chol(post_cov_scale_chol);

    double log_marg_likelihood =
        -std::log(M_PI) * (dy * n / 2.0) -
        (dy / 2.0) * log_det_prior_col +
        (dy / 2.0) * log_det_posterior_col +
        (v_prior / 2.0) * log_det_prior_cov_scale -
        (v_post / 2.0) * log_det_post_cov_scale +
        log_mvgamma((v_post / 2.0), dy) -
        log_mvgamma((v_prior / 2.0), dy);

    return log_marg_likelihood;
}

#endif