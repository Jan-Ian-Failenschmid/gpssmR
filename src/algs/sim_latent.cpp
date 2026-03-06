// [[Rcpp::depends(RcppArmadillo)]]
#include <RcppArmadillo.h>
#include "matniw_gp_class.h"
#include "imc_gp_class.h"
#include "imc_iw_class.h"
#include "gp_model_base_class.h"
#include "hsgp_class.h"
#include "sim_latent.h"
#include "linear_algebra.h"

arma::mat simulate_latent(std::unique_ptr<gp_model_base> &model_ptr,
                          const arma::mat &covariate,
                          const uint &n_time,
                          const uint &d_lat,
                          const arma::vec &t0_mean,
                          const arma::mat &t0_cov)
{
    if (auto p = dynamic_cast<imc_iw *>(model_ptr.get()))
        return sim_latent(*p, covariate, n_time, d_lat, t0_mean, t0_cov);
    if (auto p = dynamic_cast<matniw_gp_model *>(model_ptr.get()))
        return sim_latent(*p, covariate, n_time, d_lat, t0_mean, t0_cov);

    throw std::runtime_error("Unknown GP model type for sim_latent");
}

arma::mat sim_latent(imc_iw &model,
                     const arma::mat &covariate,
                     const uint &n_time,
                     const uint &d_lat,
                     const arma::vec &t0_mean,
                     const arma::mat &t0_cov)
{
    return sim_latent_exact(
        covariate,
        n_time,
        d_lat,
        std::exp(model.multiv_gp.alpha), // or model-specific hyperparameters
        std::exp(model.multiv_gp.rho),
        t0_mean,
        t0_cov,
        model.des_covar_mat, // State covariate effect matrix
        model.cov            // Dynamic error matrix
    );
}

arma::mat sim_latent(matniw_gp_model &model,
                     const arma::mat &covariate,
                     const uint &n_time,
                     const uint &d_lat,
                     const arma::vec &t0_mean,
                     const arma::mat &t0_cov)
{
    return ::sim_latent_approx( // call your existing non-exact function
        covariate,
        n_time,
        d_lat,
        model.hsgp,
        t0_mean,
        t0_cov,
        model.des_int_mat,   // State transition matrix
        model.des_covar_mat, // State covariate effect matrix
        model.cov            // Dynamic error matrix
    );
}

// Simulate latent variable
arma::mat sim_latent_approx(
    const arma::mat &covariate, // Covariate data
    const uint &n_time,         // Number of time-points
    const uint &d_lat,          // State dimensions
    hsgp_approx &hsgp,
    const arma::vec &t0_mean,   // Latent mean at t0
    const arma::mat &t0_cov,    // Latent mean at t0
    const arma::mat &trans_mat, // State transition matrix
    const arma::mat &lat_covar, // State transition matrix
    const arma::mat &dyn_cov    // Dynamic error matrix
)
{
    // Initialize latent variable matrix
    arma::mat x(d_lat, n_time);

    // Sample starting values
    x.col(0) = t0_mean +
               chol(t0_cov, "lower") *
                   arma::randn(d_lat, arma::distr_param(0.0, 1.0));

    // Precompute cholesky transforms
    arma::mat dyn_cov_chol = chol(dyn_cov, "lower"); // Precompute cholesky

    for (size_t t = 1; t < n_time; t++)
    {
        x.col(t) = trans_mat * 
            hsgp.basis_functions(x.col(t - 1)) + 
            lat_covar * covariate.col(t - 1);
        x.col(t) += dyn_cov_chol * arma::randn(d_lat,
                                               arma::distr_param(0.0, 1.0));
    }

    return x;
};

arma::mat sim_latent_exact(
    const arma::mat &covariate,
    const uint &n_time,
    const uint &d_lat,
    const double &alpha,
    const double &rho,
    const arma::vec &t0_mean, // Latent mean at t0
    const arma::mat &t0_cov,  // Latent mean at t0
    const arma::mat &lat_covar,
    const arma::mat &dyn_cov)
{
    arma::mat x_sample(d_lat, n_time);

    // std::vector<multiv_gp> multi_output_gp;
    // multi_output_gp.resize(d_lat);

    imc_gp multi_output_gp;

    // Draw first latent variable value from its prior
    x_sample.col(0) = t0_mean + chol(t0_cov, "lower") *
                                    arma::randn(d_lat, arma::distr_param(0.0, 1.0));

    // Initialize GP without training and outcome data
    multi_output_gp.update_hyperparameters(alpha, rho);
    multi_output_gp.update_sigma(dyn_cov);

    multi_output_gp.update_train_data(
        arma::mat(d_lat, 0u), arma::mat(d_lat, 0u));
    multi_output_gp.set_train_y_cov_I();
    multi_output_gp.update_test_data(x_sample.col(0));
    multi_output_gp.set_test_y_cov_I();

    // Add first prediction to GP
    // This returns a vector over test locations not dimensions
    // might make sense to transpose this
    x_sample.col(1) = multi_output_gp.make_marginal_predictions() +
                      lat_covar * covariate.col(0);

    for (size_t t = 1; t < (n_time - 1); t++)
    {
        // Use previous time points as training set
        multi_output_gp.update_train_data(
            x_sample.cols(0, t - 1),
            x_sample.cols(1, t));
        multi_output_gp.set_train_y_cov_I();
        // Use untransformed (independent GPs) as outcomes
        multi_output_gp.update_test_data(x_sample.col(t));
        x_sample.col(t + 1) = multi_output_gp.make_marginal_predictions() +
                              lat_covar * covariate.col(t);
    }
    // Return all objects in a list
    return x_sample;
};