// [[Rcpp::depends(RcppArmadillo)]]
#include <RcppArmadillo.h>
#include "pdfs.h"
#include "linear_algebra.h"
#include "niw_base_class.h"
#include "matniw_model_class.h"

matniw_model::matniw_model(
    const arma::mat &des_int_mat_const,
    const arma::mat &des_covar_mat_const,
    const arma::mat &des_int_mat_mean,
    const arma::mat &des_covar_mat_mean,
    const arma::mat &des_int_mat_col_cov,
    const arma::mat &des_covar_mat_col_cov,
    uint cov_df,
    const arma::mat &cov_scale)
    : normal_inverse_wishart_base(des_int_mat_const,
                                  des_covar_mat_const,
                                  cov_df,
                                  cov_scale)
{
    // Resize members
    phi.set_size(d_outcome, d_outcome);
    psi.set_size(d_outcome, d_predictor);
    sigma.set_size(d_predictor, d_predictor);

    post_col_cov.set_size(d_predictor, d_predictor);
    post_col_cov_chol.set_size(d_predictor, d_predictor);
    post_precision_chol.set_size(d_predictor, d_predictor);
    post_des_mat_mean.set_size(d_outcome, d_predictor);
    prior_precision_chol.set_size(d_predictor, d_predictor);
    prior_col_cov_chol.set_size(d_predictor, d_predictor);

    // If there are finite elements in condition mask, set need_cond = true
    arma::uvec fixed = arma::find_finite(des_mat_const);
    if (fixed.n_elem > 0)
        need_cond = true;
    else
        need_cond = false;

    prior_des_mat_mean =
        join_rows(des_int_mat_mean, des_covar_mat_mean);

    update_col_cov(des_int_mat_col_cov,
                   des_covar_mat_col_cov);

    // Calculate invariant prior log determinants
    log_det_prior_cov_scale = log_det_sympd_cust(cov_scale);
}

// Struct specific helper functions
// Conditionion multivariate normal for sampling from conditional prior
// and posterior
void matniw_model::cond_mv_normal(
    const arma::vec &mu,
    const arma::mat &sigma)
{
    // Conditional multivariate log likelihood
    arma::mat cond_mask = des_mat_const;

    arma::uvec fixed = arma::find_finite(cond_mask);
    arma::uvec free = arma::find_nonfinite(cond_mask);

    arma::vec mu_free = mu(free);   // mu1
    arma::vec mu_fixed = mu(fixed); // mu2

    arma::mat sigma_free = sigma(free, free);        // Sigma11
    arma::mat sigma_free_fixed = sigma(free, fixed); // Sigma12
    arma::mat sigma_fixed_free = sigma(fixed, free); // Sigma21
    arma::mat sigma_fixed = sigma(fixed, fixed);     // Sigma22

    arma::mat inv_sigma_fixed = arma::pinv(sigma_fixed);
    arma::mat sff_inv_sf = sigma_free_fixed * inv_sigma_fixed;

    arma::vec cond_vals = cond_mask.elem(fixed);
    cond_mu = mu_free + sff_inv_sf * (cond_vals - mu_fixed);

    cond_sigma = sigma_free - sff_inv_sf * sigma_fixed_free;

    chol_cond_sigma = chol(cond_sigma, "lower");
}

void matniw_model::sample_cond_mv_normal()
{
    // Why am I not using sample_matrix_normal here?
    arma::vec z(cond_mu.size(), arma::fill::randn);
    cond_sample = cond_mu + chol_cond_sigma * z;
};

void matniw_model::sample_matrix_normal(
    const arma::mat &des_mat_mean,
    const arma::mat &col_cov_chol)
{
    arma::mat cov_chol = arma::chol(cov, "lower");

    des_mat = des_mat_mean + cov_chol * des_mat.randn() * col_cov_chol.t();

    des_int_mat = des_mat.cols(0, des_int_mat.n_cols - 1);
    if (des_int_mat.n_cols < des_mat.n_cols)
    {
        des_covar_mat = des_mat.cols(des_int_mat.n_cols, des_mat.n_cols - 1);
    }
    else
    {
        des_covar_mat = arma::mat(des_mat.n_rows, 0);
    }
};

// Priors ----
void matniw_model::update_mean(
    const arma::mat &des_int_mat_mean, // Means
    const arma::mat &des_covar_mat_mean)
{
    prior_des_mat_mean = join_rows(des_int_mat_mean, des_covar_mat_mean);
};

void matniw_model::update_col_cov(
    const arma::mat &des_int_mat_col_cov,
    const arma::mat &des_covar_mat_col_cov)
{
    // This might be a cause for numerical issues if I join them together
    // as covariances and then invert them, but it should be fine.
    if (des_covar_mat_col_cov.n_elem == 0)
    {
        prior_col_cov = des_int_mat_col_cov;
    }
    else
    {
        prior_col_cov = diag_join(des_int_mat_col_cov,
                                  des_covar_mat_col_cov);
    }

    stabalized_inv(prior_col_cov, prior_precision,
                   prior_col_cov);

    prior_col_cov_chol = chol(prior_col_cov, "lower");
}

void matniw_model::sample_prior()
{
    cov = arma::iwishrnd(cov_scale, cov_df); // Sample covariance from prior
    if (need_cond)
    {
        cond_mv_normal(
            prior_des_mat_mean.as_col(),   // Prior mean
            arma::kron(prior_col_cov, cov) // Prior covariance
        );
        sample_cond_mv_normal();
        fill_sample_into_des_mat(cond_sample);
    }
    else
    {
        sample_matrix_normal(prior_des_mat_mean, prior_col_cov_chol);
    }
}

void matniw_model::sample_prior_without_cov()
{
    if (need_cond)
    {
        cond_mv_normal(
            prior_des_mat_mean.as_col(),   // Prior mean
            arma::kron(prior_col_cov, cov) // Prior covariance
        );
        sample_cond_mv_normal();
        fill_sample_into_des_mat(cond_sample);
    }
    else
    {
        sample_matrix_normal(prior_des_mat_mean, prior_col_cov_chol);
    }
}

void matniw_model::calc_posterior_parameters()
{
    // Caluculate sum of outer-products
    // Compute statistics needed to sample from Q
    // EQ: 9  in Svensson et al. 2016 (Note: Zeta appear to be going from
    // 1 to T - 1 and z from 0 to T-2 in zero indexing in Svenson but not in
    // Wills)
    // Also EQ 38 b & c in Wills et al. 2012
    phi = outcome * outcome.t();
    psi = outcome * predictor.t();
    sigma = predictor * predictor.t();

    // post_precision_chol = chol_rank_n_update(prior_precision_chol, 1,
    //                                          predictor);
    // post_col_cov_chol = arma::inv(arma::trimatl(post_precision_chol)).t();
    // post_col_cov = post_col_cov_chol * post_col_cov_chol.t();

    // Posterior col_cov col_cov_post
    post_col_cov = arma::inv_sympd(
        sigma + prior_precision); // Sigma hat in Wills et al. 2012
    post_col_cov = 0.5 * (post_col_cov + post_col_cov.t());
    post_col_cov_chol = arma::chol(post_col_cov, "lower");

    // EQ 41 in Wills et al. 2012
    post_des_mat_mean =
        (prior_des_mat_mean * prior_precision + psi) * post_col_cov;
}

void matniw_model::sample_joint_posterior()
{
    // Final calculation of post_cov_scale moved here du to differences
    // in calculation for sample_posterior_cov_cond_des_mat()
    // arma::mat pi_k = phi +
    //                  prior_des_mat_mean * prior_precision *
    //                      prior_des_mat_mean.t() -
    //                  post_des_mat_mean * psi.t();

    // pi_k = 0.5 * (pi_k + pi_k.t()); // Make sure this is symmetric!

    arma::mat Vn_inv = sigma + prior_precision;

    arma::mat quad =
        post_des_mat_mean *
        Vn_inv *
        post_des_mat_mean.t();

    arma::mat pi_k =
        phi +
        prior_des_mat_mean * prior_precision *
            prior_des_mat_mean.t() -
        quad;
    pi_k = 0.5 * (pi_k + pi_k.t()); // Make sure this is symmetric!

    post_cov_scale = cov_scale + pi_k;

    // Eq: 12 in Svensson et al. 2016 and EQ 42 in Wills et al. 2012
    // Sub post_cov_scale
    cov = arma::iwishrnd(
        post_cov_scale, n_time + cov_df);

    // Eq: 11 in Svensson et al. 2016
    if (need_cond)
    {
        cond_mv_normal(
            post_des_mat_mean.as_col(),   // Posterior mean
            arma::kron(post_col_cov, cov) // Posterior covariance
        );
        sample_cond_mv_normal(); // Samples to cond_sample
        fill_sample_into_des_mat(cond_sample);
    }
    else
    {
        sample_matrix_normal(post_des_mat_mean, post_col_cov_chol);
    }
};

void matniw_model::sample_posterior_des_mat_cond_cov()
{
    if (need_cond)
    {
        // Eq: 11 in Svensson et al. 2016
        cond_mv_normal(
            post_des_mat_mean.as_col(),   // Posterior mean
            arma::kron(post_col_cov, cov) // Posterior covariance
        );
        sample_cond_mv_normal();
        fill_sample_into_des_mat(cond_sample);
    }
    else
    {
        sample_matrix_normal(post_des_mat_mean, post_col_cov_chol);
    }
}

double matniw_model::log_marginal_likelihood()
{
    // Look into this
    arma::mat Vn_inv = sigma + prior_precision;

    arma::mat quad =
        post_des_mat_mean *
        Vn_inv *
        post_des_mat_mean.t();

    arma::mat pi_k =
        phi +
        prior_des_mat_mean * prior_precision *
            prior_des_mat_mean.t() -
        quad;

    // Perhaps move to update posterior parameters
    // arma::mat pi_k = phi +
    //                  prior_des_mat_mean * prior_precision *
    //                      prior_des_mat_mean.t() -
    //                  post_des_mat_mean * psi.t();
    pi_k = 0.5 * (pi_k + pi_k.t()); // Make sure this is symmetric!
    post_cov_scale = cov_scale + pi_k;

    log_det_Lambda_nod = log_det_chol(prior_col_cov_chol);

    log_det_Lambda_n = log_det_chol(post_col_cov_chol);

    log_det_post_cov_scale = log_det_sympd(
        0.5 * (post_cov_scale + post_cov_scale.t()));

    double log_marg_likelihood =
        -std::log(M_PI) * (d_outcome * n_time / 2.0) -
        log_det_Lambda_nod * (d_outcome / 2.0) +
        log_det_Lambda_n * (d_outcome / 2.0) +
        log_det_prior_cov_scale * (cov_df / 2.0) -
        log_det_post_cov_scale * ((cov_df + n_time) / 2.0) -
        log_mvgamma(cov_df / 2.0, d_outcome) +
        log_mvgamma((cov_df + n_time) / 2.0, d_outcome);

    return log_marg_likelihood;
}
