// [[Rcpp::depends(RcppArmadillo)]]

#include <RcppArmadillo.h>
#include "pgas.h"
#include "imc_gp_struct.h"
#include "hsgp_struct.h"
#include "base_structs.h"
#include "derived_structs.h"
#include "resampling.h"

// Measurement model -----
// Measurement model helper function used in PGAS to predict measurements
arma::mat meas_model(
    const arma::mat &x,         // Matrix of states D * T or D * N
    const arma::vec &covariate, // Vector
    const arma::mat &des_mat,   // Design Matrix
    const arma::mat &meas_covar // Covariate effect
)
{
    arma::mat out = des_mat * x;

    if (covariate.n_rows > 0)
    {
        out.each_col() += meas_covar * covariate;
    }

    return out;
}

// Transition model -----
// Transition model helper function used in PGAS to predict future states
arma::mat transit_model(
    const arma::mat &x,         // Vector of states D * 1 or
    const arma::vec &covariate, // Vector of covariates D * 1
    const arma::mat &trans_mat, // Transition Matrix
    const arma::mat &lat_covar  // Covariate
)
{
    arma::mat out = trans_mat * x;
    if (covariate.n_rows > 0)
    {
        out.each_col() += lat_covar * covariate;
    }

    return out;
}

// PGAS markov kernel used in the gibbs sampler to sample state trajectories
arma::mat pgas(
    const arma::mat &y,
    const arma::mat &covariate_dyn,
    const arma::mat &covariate_meas,
    const int &n_particles,
    const int &n_time,
    const int &d_lat,
    const hsgp_approx &hsgp,
    const arma::mat &x_ref,
    const arma::vec &t0_mean,
    const arma::mat &t0_cov,
    const arma::mat &trans_mat,
    const arma::mat &lat_covar,
    const arma::mat &dyn_cov,
    const arma::mat &des_mat,
    const arma::mat &meas_covar,
    const arma::mat &meas_cov)
{
    // PGAS kernel from Lindsten et al. 2014 modified to assumo a Markovian
    // state progression as in Svenson et al. 2016.
    hsgp_approx hsgp_local = hsgp;
    // Initialize a cube to hold the particle values over all particles N
    // the state dimensions D and the time-points T
    arma::cube x(d_lat, n_particles, n_time);

    // Initialize weight matrix
    // Setting weights to zero here, because they get softmaxed at each iteration
    // of the for-loop below
    arma::mat weights(n_time, n_particles, arma::fill::zeros);

    // Initialize ancestor matrtix
    arma::umat ancestors(n_time - 1, n_particles, arma::fill::zeros);

    // Initialize noise matrix
    arma::mat noise(d_lat, n_particles - 1);

    // Initialize matrix to hold temp predictions
    arma::mat x_pred(d_lat, n_particles, arma::fill::zeros);
    arma::mat x_pred_resampled(d_lat, n_particles, arma::fill::zeros);
    arma::mat y_pred(des_mat.n_rows, n_particles, arma::fill::zeros);
    arma::mat residuals(des_mat.n_rows, n_particles, arma::fill::zeros);

    // Vector holding temp weights
    arma::rowvec weights_n(n_particles, arma::fill::zeros);

    // Precompute parameter transformations
    arma::mat dyn_cov_inv = inv_sympd(dyn_cov);      // Precompute matrix inverse
    arma::mat dyn_cov_chol = chol(dyn_cov, "lower"); // Precompute cholesky
    arma::mat meas_cov_inv = inv_sympd(meas_cov);    // Precompute matrix inverse

    // Precompute parameters for proposal distribution
    // Proposal function adapted from:
    // Snyder, "Particle filters, the “optimal” proposal and high-dimensional systems"
    arma::mat t0_cov_inv = inv_sympd(t0_cov);
    arma::mat t0_kalman_gain = t0_cov * des_mat.t() *
                               inv_sympd(des_mat * t0_cov * des_mat.t() +
                                         meas_cov);
    arma::vec t0_prop_mean = t0_mean +
                             t0_kalman_gain * (y.col(0) -
                                               meas_model(t0_mean,
                                                          covariate_meas.col(0),
                                                          des_mat, meas_covar));

    arma::mat t0_prop_cov = t0_cov - t0_kalman_gain * des_mat * t0_cov;
    arma::mat t0_prop_cov_inv = inv_sympd(t0_prop_cov);
    arma::mat t0_prop_cov_chol = chol(t0_prop_cov, "lower");

    // Analytic covariance of the weights distribution Snyder eq.: 16
    // From this equation it also follows that particles with the same mean
    // (predicted value/x_pred_resampled) have the same weight. Since, all
    // particles have the same mean at t0 (t0_mean) calculating the weights
    // can be skipped.
    arma::mat weights_cov = des_mat * dyn_cov * des_mat.t() + meas_cov;
    arma::mat weights_cov_chol = chol(weights_cov, "lower");

    arma::mat dyn_kalman_gain = dyn_cov * des_mat.t() *
                                inv_sympd(des_mat * dyn_cov * des_mat.t() +
                                          meas_cov);
    arma::mat prop_mean(d_lat, n_particles);
    arma::mat prop_cov = dyn_cov - dyn_kalman_gain * des_mat * dyn_cov;
    arma::mat prop_cov_inv = inv_sympd(prop_cov);
    arma::mat prop_cov_chol = chol(prop_cov, "lower");

    // Sample particle starting value from distribution of x_1
    // Step 1 of Algorithm 2 in Svensson et al. 2016
    noise.randn();
    x.slice(0).submat(0, 0, d_lat - 1, n_particles - 2) =
        t0_prop_cov_chol * noise;
    x.slice(0).each_col() += t0_prop_mean;

    // Loop the remaining steps over all time points up to T
    // Step 3 of Algorithm 2 in Svensson et al. 2016

    // Set starting value of the N_th particle to reference value
    // Step 2 of Algorithm 2 in Svensson et al. 2016
    x.slice(0).col(n_particles - 1) = x_ref.col(0);

    for (size_t t = 0; t < n_time; t++)
    {
        if (t >= 1)
        {
            // Systematically resample ancestor indices according to importance
            // weights
            // Step 5 of Algorithm 2 in Svensson et al. 2016
            ancestors.row(t - 1).subvec(0, n_particles - 2) =
                systematic_resampling(weights.row(t - 1), n_particles - 1);

            // Precalculate predictions of all previous particle states and store in
            // x_pred for later use
            x_pred = transit_model(
                hsgp_local.basis_functions(x.slice(t - 1)),
                covariate_dyn.col(t - 1), trans_mat, lat_covar);

            // Resample predictions according to ancestor indices
            x_pred_resampled.cols(0, n_particles - 2) =
                x_pred.cols(ancestors.row(t - 1).subvec(0, (n_particles - 2)));

            // Make observation prediction based on one-step state forecast
            y_pred.cols(0, n_particles - 2) = meas_model(
                x_pred_resampled.cols(0, n_particles - 2), covariate_meas.col(t),
                des_mat, meas_covar);
            residuals.cols(0, n_particles - 2) = -y_pred.cols(0, n_particles - 2);
            residuals.cols(0, n_particles - 2).each_col() += y.col(t);
            prop_mean.cols(0, n_particles - 2) =
                x_pred_resampled.cols(0, n_particles - 2) + dyn_kalman_gain * residuals.cols(0, n_particles - 2);

            // Step 6 of Algorithm 2 in Svensson et al. 2016
            // Select predictions from x_pred_i for t + 1 according to the ancestral
            // indices stored in a.row(t - 1)
            // Sample a new state values around the prediction in a vectorised
            // fashion and store in x for the next time point
            noise.randn();
            // x.slice(t).submat(0, 0, d_lat - 1, n_particles - 2) =
            //     x_pred.cols(ancestors.row(t - 1).subvec(0, (n_particles - 2))) + dyn_cov_chol * noise;
            x.slice(t).submat(0, 0, d_lat - 1, n_particles - 2) =
                prop_mean.cols(0, n_particles - 2) + prop_cov_chol * noise;

            // Set last particle state to state from the reference path
            // Step 7 of Algorithm 2 in Svensson et al. 2016
            x.slice(t).col(n_particles - 1) = x_ref.col(t); // X^N_(t+1)

            // Sample an ancestor index for the reference particle
            // Step 8 of Algorithm 2 in Svensson et al. 2016
            // Calculate the density of predicting the reference particle from each
            weights_n = arma::trans(mat_logdnorm(
                arma::vec(x.slice(t).col(n_particles - 1)),
                x_pred, dyn_cov_chol));

            // You can ignore the denominator here since it cancells accorss all
            // weights
            // Elementwise multiply weights and normal densities
            weights_n += log(weights.row(t - 1));
            softmax(weights_n); // Normalize weights

            // Resample the ancestral path for the reference particle from the weights
            ancestors.row(t - 1)(n_particles - 1) = systematic_resampling(
                weights_n, 1)(0);

            // Add predicted value for the reference particle to the resampled matrix
            // for weight calculation
            x_pred_resampled.col(n_particles - 1) =
                x_pred.col(ancestors.row(t - 1)(n_particles - 1));

            // Calculate weights
            weights.row(t) = arma::trans(mat_logdnorm(arma::vec(y.col(t)),
                                                      des_mat * x_pred_resampled, weights_cov_chol));
        }
        softmax(weights.row(t));
    }

    // Initialize a matrix holding the output sample from the invariant
    // state distribution
    arma::mat x_out(d_lat, n_time, arma::fill::zeros);

    // Sample an index from the weights at the last time point
    // Step 11 of Algorithm 2 in Svensson et al. 2016
    uint star = systematic_resampling(weights.row(n_time - 1), 1)(0);

    // Trace the ancestral path of the selected index particle and combine
    // all its ancestors into the output sample stored in x_out
    // Step 9 of Algorithm 2 in Svensson et al. 2016
    x_out.col((n_time - 1)) = x.slice((n_time - 1)).col(star);
    for (size_t i = 1; i < n_time; i++)
    {
        star = ancestors((n_time - 1) - i, star);
        x_out.col((n_time - 1) - i) = x.slice((n_time - 1) - i).col(star);
    }

    // Return new sample from the invariant state distribution
    return x_out;
}

arma::mat pgas(
    const arma::mat &y,
    const arma::mat &covariate_dyn,
    const arma::mat &covariate_meas,
    const int &n_particles,
    const int &n_time,
    const int &d_lat,
    const imc_gp &gp,
    const arma::mat &x_ref,
    const arma::vec &t0_mean,
    const arma::mat &t0_cov,
    const arma::mat &lat_covar,
    const arma::mat &dyn_cov,
    const arma::mat &des_mat,
    const arma::mat &meas_covar,
    const arma::mat &meas_cov)
{
    // PGAS kernel from Lindsten et al. 2014 modified
    int d_meas = y.n_rows;

    // Initialize a cube to hold the particle values over all particles N
    // the state dimensions D and the time-points T
    arma::cube x(d_lat, n_particles, n_time);

    // Initialize weight matrix
    // Setting weights to zero here, because they get softmaxed at each iteration
    // of the for-loop below
    arma::mat weights(n_time, n_particles, arma::fill::zeros);

    // Initialize ancestor matrtix
    arma::umat ancestors(n_time - 1, n_particles, arma::fill::zeros);

    // Precalculate parmaters for proposal distribution
    // Precompute parameters for proposal distribution
    // Proposal function adapted from:
    // Snyder, "Particle filters, the “optimal” proposal and high-dimensional systems"
    arma::mat meas_cov_inv = inv(meas_cov); // Precompute matrix inverse
    arma::mat meas_cov_chol = chol(meas_cov, "lower");

    arma::mat t0_cov_inv = inv_sympd(t0_cov);
    arma::mat t0_kalman_gain = t0_cov * des_mat.t() *
                               inv_sympd(des_mat * t0_cov * des_mat.t() +
                                         meas_cov);
    arma::vec t0_prop_mean = t0_mean +
                             t0_kalman_gain * (y.col(0) -
                                               meas_model(t0_mean,
                                                          covariate_meas.col(0),
                                                          des_mat, meas_covar));

    arma::mat t0_prop_cov = t0_cov - t0_kalman_gain * des_mat * t0_cov;
    arma::mat t0_prop_cov_inv = inv_sympd(t0_prop_cov);
    arma::mat t0_prop_cov_chol = chol(t0_prop_cov, "lower");

    // Analytic covariance of the weights distribution Snyder eq.: 16
    // From this equation it also follows that particles with the same mean
    // (predicted value/x_pred_resampled) have the same weight. Since, all
    // particles have the same mean at t0 (t0_mean) calculating the weights
    // can be skipped.

    arma::mat dyn_cov_k(d_lat, d_lat);
    arma::mat dyn_cov_k_chol(d_lat, d_lat);

    arma::mat weights_cov(d_meas, d_meas);
    arma::mat weights_cov_chol(d_meas, d_meas);
    arma::mat weights_cov_inv(d_meas, d_meas);
    arma::mat dyn_kalman_gain(d_lat, d_meas);

    arma::vec prop_mean(d_lat);
    arma::mat prop_cov(d_lat, d_lat);
    arma::mat prop_cov_inv(d_lat, d_lat);
    arma::mat prop_cov_chol(d_lat, d_lat);

    arma::vec y_pred(d_meas);
    arma::vec residuals(d_lat);

    // Initialize particles ------
    // Sample particle starting value from distribution of x_1
    x.slice(0).submat(0, 0, d_lat - 1, n_particles - 2) =
        t0_prop_cov_chol *
        arma::randn(d_lat, n_particles - 1, arma::distr_param(0.0, 1.0));
    x.slice(0).each_col() += t0_prop_mean;

    // Set starting value of the N_th particle to reference value
    x.slice(0).col(n_particles - 1) = x_ref.col(0);

    std::vector<imc_gp> multi_output_gp;
    multi_output_gp.resize(n_particles);

    for (size_t i = 0; i < n_particles; i++)
    {
        multi_output_gp[i].update_hyperparameters(gp.alpha, gp.rho);
        multi_output_gp[i].update_sigma(dyn_cov);

        multi_output_gp[i].update_train_data(
            arma::mat(d_lat, 0u),
            arma::mat(d_lat, 0u));
        multi_output_gp[i].set_train_y_cov_I();
        multi_output_gp[i].update_test_data(x.slice(0).col(i));
        multi_output_gp[i].set_test_y_cov_I();
    }

    // Loop the remaining steps over all time points up to T
    // arma::mat x_pred_i(d_lat, n_particles, arma::fill::zeros);

    // Vector holding temp weights
    arma::rowvec weights_n(n_particles, arma::fill::zeros);
    arma::mat x_ref_adj = x_ref;
    for (size_t k = 1; k < n_time; k++)
    {
        x_ref_adj.col(k) -= lat_covar * covariate_dyn.col(k - 1);
    }
    arma::vec BZ;

    for (size_t t = 0; t < n_time; t++)
    {
        if (t >= 1)
        {
            // Ancestor sampling -----
            // Standard resampling for particles 1:N-1
            ancestors.row(t - 1).subvec(0, n_particles - 2) =
                systematic_resampling(weights.row(t - 1), n_particles - 1);

            // Calculate ancestor weights for reference particle
            for (size_t i = 0; i < n_particles; i++)
            {
                // Sample ancestor index for reference particle
                if (t < n_time - 1)
                {
                    // Current test data is x_t-1
                    multi_output_gp[i].append_test_data(x_ref.cols(t, n_time - 2));
                    multi_output_gp[i].append_test_y_cov_I();
                    // Test data is x_t-1:T-1
                }
                weights_n(i) = multi_output_gp[i].test_marginal_log_likelihood(
                    x_ref_adj.cols(t, n_time - 1));
                // According to P(X_t:T | X'_0:t-1)

                // Reset test data to only x_t-1
                multi_output_gp[i].reset_test_data();
                multi_output_gp[i].reset_test_y_cov();
            }

            weights_n += log(weights.row(t - 1));
            softmax(weights_n); // Normalize weights

            ancestors.row(t - 1)(n_particles - 1) = systematic_resampling(
                weights_n, 1)(0);

            // Particle propagation -----
            // resample particles according to ancestor indices
            multi_output_gp = resample_std_vector(multi_output_gp,
                                                  ancestors.row(t - 1).t());

            for (size_t i = 0; i < n_particles; i++)
            {
                // For all particles except the reference particle
                // make predictions for x_t
                multi_output_gp[i].compute_predictive(true);
                // dyn_cov_k = arma::kron(
                //     multi_output_gp[i].pred_col_cov_chol *
                //         multi_output_gp[i].pred_col_cov_chol.t(),
                //     multi_output_gp[i].sigma);

                dyn_cov_k_chol = arma::kron(
                    multi_output_gp[i].pred_col_cov_chol,
                    multi_output_gp[i].sigma_chol);

                // weights_cov = des_mat * dyn_cov_k * des_mat.t() + meas_cov;

                weights_cov_chol = chol_rank_n_update(
                    meas_cov_chol, 1, des_mat * dyn_cov_k_chol);

                // weights_cov_inv = inv_sympd(weights_cov);
                // dyn_kalman_gain = dyn_cov_k * des_mat.t() *
                //                   inv_sympd(des_mat * dyn_cov_k * des_mat.t() +
                //                             meas_cov);
                prop_cov_chol = chol_rank_n_update(
                    dyn_cov_k_chol, -1,
                    arma::solve(
                        weights_cov_chol,
                        des_mat * dyn_cov_k_chol * dyn_cov_k_chol.t())
                        .t());

                // prop_cov = dyn_cov_k - dyn_kalman_gain * des_mat * dyn_cov_k;
                // prop_cov_inv = inv_sympd(prop_cov);
                // prop_cov_chol = chol(prop_cov, "lower");
                BZ = lat_covar * covariate_dyn.col(t - 1);
                y_pred = meas_model(
                    multi_output_gp[i].pred_mean + BZ,
                    covariate_meas.col(t),
                    des_mat, meas_covar);
                residuals = y.col(t) - y_pred;

                // prop_mean = multi_output_gp[i].pred_mean + dyn_kalman_gain *residuals;

                arma::vec v =
                    solve(trimatu(weights_cov_chol.t()),
                          solve(trimatl(weights_cov_chol), residuals));

                prop_mean = multi_output_gp[i].pred_mean + BZ +
                            dyn_cov_k_chol * dyn_cov_k_chol.t() * des_mat.t() * v;

                if (i < (n_particles - 1))
                {
                    x.slice(t).col(i) = prop_cov_chol * arma::randn(d_lat);
                    x.slice(t).col(i) += prop_mean;
                }

                weights.row(t)(i) = logdnorm(
                    y.col(t),
                    des_mat * (multi_output_gp[i].pred_mean + BZ),
                    weights_cov_chol);
            }

            // Set last particle state to state from the reference path
            x.slice(t).col(n_particles - 1) = x_ref.col(t); // X^N_(t+1)

            // Update GP posteriors to include newly sampled data ------
            // Add new predictions to particles
            for (size_t i = 0; i < n_particles; i++)
            {
                // Append X_t-1 to training data
                multi_output_gp[i].append_train_data(
                    multi_output_gp[i].test_dat,
                    x.slice(t).col(i) - BZ);
                multi_output_gp[i].append_train_y_cov_I();
                // Append new predction for x_t to the outcome data
                // Set test data to new prediction for x_t
                multi_output_gp[i].update_test_data(x.slice(t).col(i));
                multi_output_gp[i].set_test_y_cov_I();
            }
        }

        // Caluclate weights -----
        softmax(weights.row(t));
    }

    // Initialize a matrix holding the output sample from the invariant
    // state distribution
    arma::mat x_out(d_lat, n_time, arma::fill::zeros);

    // Sample an index from the weights at the last time point
    uint star = systematic_resampling(weights.row(n_time - 1), 1)(0);

    // Trace the ancestral path of the selected index particle and combine
    // all its ancestors into the output sample stored in x_out
    x_out.col((n_time - 1)) = x.slice((n_time - 1)).col(star);
    for (size_t i = 1; i < n_time; i++)
    {
        star = ancestors((n_time - 1) - i, star);
        x_out.col((n_time - 1) - i) = x.slice((n_time - 1) - i).col(star);
    }

    // Return new sample from the invariant state distribution
    return x_out;
}
