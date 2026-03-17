// Dependencies       ----
// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::depends(RcppProgress)]]
#include <RcppArmadillo.h>
#include <progress.hpp>
#include <progress_bar.hpp>
#include <cmath> // For M_PI
#include <rcpptimer.h>

#include "linear_algebra.h"
#include "resampling.h"
#include "hsgp_struct.h"
#include "imc_gp_struct.h"
#include "mh_kernel.h"
#include "pgas.h"
#include "sim_latent.h"
#include "base_structs.h"
#include "derived_structs.h"
#include "main_helper.h"
#include "timer.h"

using namespace Rcpp;

// [[Rcpp::export]]
arma::mat gpssm_sample(

    const uint &n_iter,     // Number of MCMC samples to be drawn
    const uint &n_warm_up,  // Number of warm up samples to be drawn
    const uint &n_thin,     //  Prethinning during sampling
    const int &n_particles, // Number of particles

    const uint &n_time,              // Number of time-points
    const int &d_lat,                // Number of time-points
    const int &d_obs,                // Number of time-points
    arma::mat y,                     // Observation matrix y
    arma::mat x,                     // Latent variable initialization
    const arma::mat &covariate_dyn,  // Observation matrix y
    const arma::mat &covariate_meas, // Observation matrix y

    const arma::vec &t0_mean, // Latent mean at t0
    const arma::mat &t0_cov,  // Latent mean at t0

    const arma::mat &basis_fun_index, // Basis function index
    const arma::vec &boundry_factor,  // Boundry factor
    const Rcpp::Function &dprior,     // Hyperparameter log prior
    const Rcpp::Function &rprior,     // Hyperparameter rng

    const arma::mat dyn_design_mat_const,  // Internal matrix constraint
    arma::mat dyn_design_mat_mean,         // Internal matrix mean
    const arma::mat dyn_covar_mat_const,   // Covariate matrix mean
    const arma::mat dyn_covar_mat_mean,    // Covariate matrix mean
    const arma::mat dyn_covar_mat_col_cov, // Covariate matrix covariance
    const uint &dyn_cov_df,                // Prior df dyn_cov
    const arma::mat &dyn_cov_scale,        // Prior scale dyn_cov

    const arma::mat meas_design_mat_const,    // Internal matrix constraint
    const arma::mat meas_design_mat_mean,     // Internal matrix mean
    const arma::mat meas_design_mat_col_cov,  // Internal matrix covariance
    const arma::vec meas_design_mat_mean_alt, // Alternative prior formulation
    const arma::mat meas_design_mat_cov_alt,  // Alternative prior formulation
    const arma::mat meas_covar_mat_const,     // Covariate matrix mean
    const arma::mat meas_covar_mat_mean,      // Covariate matrix mean
    const arma::mat meas_covar_mat_col_cov,   // Covariate matrix covariance
    const arma::vec meas_covar_mat_mean_alt,  // Alternative prior formulation
    const arma::mat meas_covar_mat_cov_alt,   // Alternative prior formulation
    const uint &meas_cov_df,                  // Prior df dyn_cov
    const arma::mat &meas_cov_scale,          // Prior scale dyn_cov
    const bool uses_alt,                      // Alternative priors

    const uint &mh_rep,
    const uint &pg_rep,

    const uint mh_adapt_start,
    bool exact = false,
    bool post_pred = false,
    bool disp_prog = true)
{
    // Total iterations
    uint n_total = (n_iter + n_warm_up) * n_thin;
    timer.reset();

    // ---------------------------------
    // Set-up
    // ---------------------------------

    timer.tic("setup");

    // Initialize progress bar
    Progress p(n_total, disp_prog);

    // Initialize MH --------
    arma::vec hyperparameters(2);
    hyperparameters = Rcpp::as<arma::vec>(rprior());
    mh_kernel rw_mh(hyperparameters, dprior, n_warm_up, mh_adapt_start);

    // Extract predictors for the dynamic model
    arma::span sp_pred(0, n_time - 2);
    arma::span sp_out(1, n_time - 1);
    arma::mat x_out = x.cols(sp_out);
    arma::mat x_pred = x.cols(sp_pred);
    arma::mat covariate_pred = covariate_dyn.cols(sp_pred);

    // Likelihood parameters
    arma::mat y_mean(d_obs, n_time, arma::fill::zeros);
    arma::mat y_cov_chol = identity(n_time);
    arma::mat x_mean(d_lat, n_time - 1, arma::fill::zeros);
    arma::mat x_cov_chol = identity(n_time - 1);

    // Measurement model parameters
    arma::mat meas_design_mat_cov_chol = chol(meas_design_mat_cov_alt, "lower");
    arma::mat meas_covar_mat_cov_chol = chol(meas_covar_mat_cov_alt, "lower");
    arma::mat meas_cov_scale_chol = chol(meas_cov_scale, "lower");

    mvn_covar_wrapper meas_model_wrapper(
        &x, &covariate_meas,
        &meas_design_mat_const, &meas_covar_mat_const,
        &meas_design_mat_mean_alt, &meas_covar_mat_mean_alt,
        &meas_design_mat_cov_chol, &meas_covar_mat_cov_chol);

    // Dynamic model parameters
    // Define GP to pass parameters to dynamic model
    std::unique_ptr<gp_base> gp;
    if (exact)
    {
        gp = std::make_unique<imc_gp>();
        dyn_design_mat_mean.set_size(d_lat, n_time - 1);
    }
    else
    {
        gp = std::make_unique<hsgp_approx>(basis_fun_index,
                                           boundry_factor);
        dyn_design_mat_mean.set_size(d_lat, basis_fun_index.n_rows);
    }
    gp->set_hyperparameters(
        std::exp(hyperparameters[0]), std::exp(hyperparameters[1]));
    gp->update_predictor(x_pred);

    arma::mat dyn_covar_mat_cov_chol = chol(dyn_covar_mat_col_cov, "lower");
    arma::mat dyn_cov_scale_chol = chol(dyn_cov_scale, "lower");
    dyn_design_mat_mean.zeros();

    mn_covar_wrapper dyn_model_wrapper(
        gp->get_predictor_ptr(), &covariate_pred,
        &dyn_design_mat_mean, &dyn_covar_mat_mean,
        gp->get_cov_chol_ptr(), &dyn_covar_mat_cov_chol);

    // Initialize models
    mvn_iw_model_ meas_model = init_mvn_iw_model(
        y,
        y_mean,
        meas_model_wrapper,
        meas_cov_scale_chol,
        meas_cov_df);

    mn_iw_model_ dyn_model = init_mn_iw_model(
        x_out,
        x_mean,
        x_cov_chol,
        dyn_model_wrapper,
        dyn_cov_scale_chol,
        dyn_cov_df);

    // Lambda functions
    arma::vec par_vec;
    uint par_size;
    uint pos = 0;
    auto append = [&](const auto &v)
    {
        if (!v.n_elem)
            return;

        par_vec.subvec(pos, pos + v.n_elem - 1) = arma::vectorise(v);
        pos += v.n_elem;
    };

    //   auto set_gp_hyperpars = [&](const arma::vec &p)
    //   {
    //     gp_model->set_hyperparameters(std::exp(p[0]), std::exp(p[1]));
    //   };

    // Hyperparameters need to be set before before data
    // Set initial parameter values -------
    dyn_model.calc_posterior_parameters();
    dyn_model.sample_posterior();

    // meas_model.set_data(meas_data, meas_data_mean);
    meas_model.sample_prior();
    for (size_t i = 0; i < 1000; i++)
    {
        meas_model.calc_posterior_parameters();
        meas_model.sample_posterior();
    }

    // Initialize latent vriable -------
    // if (pg_rep > 0)
    // {
    //     if (exact)
    //     {
    //         x = sim_latent(
    //             covariate_dyn, // Covariate data
    //             n_time,        // Number of time-points
    //             d_lat,         // State dimensions
    //             *gp_exact,
    //             t0_mean,
    //             t0_cov,
    //             mn_chain->mn2->get_param(), // State transition matrix
    //             dyn_model.iw->get_cov()     // Dynamic error matrix
    //         );
    //     }
    //     else
    //     {
    //         x = sim_latent(
    //             covariate_dyn, // Covariate data
    //             n_time,        // Number of time-points
    //             d_lat,         // State dimensions
    //             *gp_approx,
    //             t0_mean,
    //             t0_cov,
    //             mn_chain->mn1->get_param(),
    //             mn_chain->mn2->get_param(), // State transition matrix
    //             dyn_model.iw->get_cov()     // Dynamic error matrix
    //         );
    //     }
    // }

    run_sim_latent(
        x, covariate_dyn,
        *gp, dyn_model, dyn_model_wrapper,
        t0_mean, t0_cov,
        pg_rep);

    x_out = x.cols(sp_out);
    x_pred = x.cols(sp_pred);
    update_model_predictor(x_pred, *gp, dyn_model, dyn_model_wrapper);
    meas_model_wrapper.combine_data();

    // Initialize temparary and output matreces
    arma::mat gp_sample(d_lat, n_time - 1);
    arma::mat y_post_pred(d_obs, n_time);
    arma::mat samples;
    timer.toc("setup");
    // ---------------------------------
    // MCMC Sampling loop
    // ---------------------------------
    for (size_t k = 0; k < n_total; k++)
    {

        // Increment progress bar
        if (Progress::check_abort())
        {
            return arma::mat(1, 1, arma::fill::zeros);
        }
        if (k % 10 == 0 && disp_prog)
        {
            p.increment(10);
        }
        timer.tic("pgas");
        // Sample state smoothing -------
        // for (size_t i = 0; i < pg_rep; i++)
        // {
        //     if (exact)
        //     {
        //         x = pgas(
        //             y,
        //             covariate_dyn,
        //             covariate_meas,
        //             n_particles,
        //             n_time,
        //             d_lat,
        //             *gp_exact,
        //             x,
        //             t0_mean,
        //             t0_cov,
        //             mn_chain->mn2->get_param(),
        //             dyn_model.iw->get_cov(),
        //             mvn_wrapper.get_pred_param(),
        //             mvn_wrapper.get_covar_param(),
        //             meas_model.iw->get_cov());
        //     }
        //     else
        //     {
        //         x = pgas(
        //             y,
        //             covariate_dyn,
        //             covariate_meas,
        //             n_particles,
        //             n_time,
        //             d_lat,
        //             &gp,
        //             x,
        //             t0_mean,
        //             t0_cov,
        //             mn_chain->mn1->get_param(),
        //             mn_chain->mn2->get_param(),
        //             dyn_model.iw->get_cov(),
        //             mvn_wrapper.get_pred_param(),
        //             mvn_wrapper.get_covar_param(),
        //             meas_model.iw->get_cov());
        //     }
        // }
        run_pgas(
            y, x, covariate_dyn, covariate_meas,
            n_particles,
            *gp, dyn_model, dyn_model_wrapper,
            meas_model, meas_model_wrapper,
            t0_mean, t0_cov,
            pg_rep);
        timer.tic("pgas_outer");

        x_out = x.cols(sp_out);
        x_pred = x.cols(sp_pred);
        update_model_predictor(x_pred, *gp, dyn_model, dyn_model_wrapper);
        meas_model_wrapper.combine_data();
        timer.toc("pgas_outer");

        timer.toc("pgas");

        // mvn_wrapper.set_data();

        // Sample measurement model parameters -------
        timer.tic("meas_model_posterior");
        meas_model.calc_posterior_parameters();
        meas_model.sample_posterior();
        timer.toc("meas_model_posterior");

        // Sample hyperparameters using MH-within-Gibbs -------
        // Marginalized MH-within-gibbs sampling for Hyperparameters
        // Start by calculating the log_marginal_likelihood of the last
        // set of hyperparameters for the new data
        timer.tic("mh");
        // set_gp_hyperpars(rw_mh.par);
        // dyn_model.set_data(dyn_data, dyn_data_mean, identity(n_time - 1));
        // dyn_model.calc_posterior_parameters();
        // rw_mh.log_lik = dyn_model.log_marginal_likelihood();

        update_model_hyperparameters(rw_mh.par, *gp, dyn_model,
                                     dyn_model_wrapper);
        // dyn_model.calc_posterior_parameters();
        rw_mh.log_lik = dyn_model.log_marginal_likelihood();

        // MH - step
        rw_mh.advance_iter();
        for (size_t i = 0; i < mh_rep; i++)
        {
            timer.tic("mh.make_prop");
            rw_mh.make_proposal(); // Make proposal
            timer.toc("mh.make_prop");
            try
            {
                timer.tic("mh.set_hyperpars1");
                update_model_hyperparameters(rw_mh.prop_par, *gp, dyn_model,
                                             dyn_model_wrapper);
                timer.toc("mh.set_hyperpars1");
                timer.tic("mh.calc_pars");
                // dyn_model.calc_posterior_parameters();
                timer.toc("mh.calc_pars");
                timer.tic("mh.calc_ll");
                rw_mh.proposal_log_lik = dyn_model.log_marginal_likelihood();
                timer.toc("mh.calc_ll");
            }
            catch (const std::exception &e)
            {
                Rcpp::Rcout << "Warning: The current Metropolis-Hastings proposal is about to be rejectected, because a parameter is outside of the valid range." << std::endl;
                Rcpp::Rcout << "Exception caught in MH proposal: " << e.what() << std::endl;
                Rcpp::Rcout << std::exp(rw_mh.prop_par[0]) << std::endl;
                Rcpp::Rcout << std::exp(rw_mh.prop_par[1]) << std::endl;
                rw_mh.proposal_log_lik = -std::numeric_limits<double>::max();
            }
            timer.tic("mh.make_step");
            rw_mh.mh_step(); // Accept or reject proposal
            timer.tic("mh.make_step");
        }
        rw_mh.tune_proposal(); // Tune proposal based on acceptance ratio

        // // Update hyperparameters
        timer.tic("mh.set_hyperpars2");
        update_model_hyperparameters(rw_mh.par, *gp, dyn_model,
                                     dyn_model_wrapper);
        timer.toc("mh.set_hyperpars2");
        timer.toc("mh");

        // Sample dynamic model parameters -------
        timer.tic("dyn_model_posterior");
        dyn_model.calc_posterior_parameters();
        dyn_model.sample_posterior();
        timer.toc("dyn_model_posterior");
        // Sample dynamic model parameters -------
        gp_sample = dyn_model_wrapper.get_pred_param() *
                    (*gp->get_predictor_ptr());
        y_post_pred = meas_model.get_param() * meas_model_wrapper.combined_data;
        y_post_pred += chol(meas_model.get_cov(), "lower") *
                       arma::randn(d_obs, n_time, arma::distr_param(0.0, 1.0));

        // ---------------------------------
        // Save samples to output
        // ---------------------------------

        // Count number of parameters
        if (k == 0)
        {
            par_size = 0;
            if (post_pred)
                par_size += y_post_pred.n_elem;
            par_size += x.n_elem;
            par_size += gp_sample.n_elem;
            par_size += rw_mh.par.n_elem;
            if (!exact)
                par_size += dyn_model_wrapper.get_pred_param().n_elem;
            par_size += dyn_model_wrapper.get_covar_param().n_elem;
            par_size += dyn_model.get_cov().n_elem;
            par_size += meas_model_wrapper.get_pred_param().n_elem;
            par_size += meas_model_wrapper.get_covar_param().n_elem;
            par_size += meas_model.get_cov().n_elem;

            par_vec.set_size(par_size);
            samples.set_size(n_total / n_thin, par_size);
        }
        // Fill parameter vector
        if (k % n_thin == 0)
        {
            pos = 0;
            if (post_pred)
            {
                append(y_post_pred);
            }
            append(x);
            append(gp_sample);
            append(arma::vec(arma::exp(rw_mh.par)));
            if (!exact)
                append(dyn_model_wrapper.get_pred_param());
            append(dyn_model_wrapper.get_covar_param());
            append(dyn_model.get_cov());
            append(meas_model_wrapper.get_pred_param());
            append(meas_model_wrapper.get_covar_param());
            append(meas_model.get_cov());

            // Store in samples matrix
            samples.row(k / n_thin) = par_vec.t();
        }
    }
    timer.stop();
    // Return samples matrix
    return samples;
};

// -----------------------------------------------------------------------------
// [[Rcpp::export]]
arma::mat gpssm_prior_sample(

    const uint &n_iter, // Number of MCMC samples to be drawn

    const int &n_time, // Number of time-points
    const int &d_lat,  // Number of time-points
    const int &d_obs,  // Number of time-points

    const arma::mat &covariate_dyn,  // Observation matrix y
    const arma::mat &covariate_meas, // Observation matrix y

    const arma::vec &t0_mean, // Latent mean at t0
    const arma::mat &t0_cov,  // Latent mean at t0

    const arma::mat &basis_fun_index, // Basis function index
    const arma::vec &boundry_factor,  // Boundry factor
    const Rcpp::Function &rprior,     // Hyperparameter rng

    const arma::mat dyn_design_mat_const,  // Internal matrix constraint
    arma::mat dyn_design_mat_mean,   // Internal matrix mean
    const arma::mat dyn_covar_mat_const,   // Covariate matrix mean
    const arma::mat dyn_covar_mat_mean,    // Covariate matrix mean
    const arma::mat dyn_covar_mat_col_cov, // Covariate matrix covariance
    const uint &dyn_cov_df,                // Prior df dyn_cov
    const arma::mat &dyn_cov_scale,        // Prior scale dyn_cov

    const arma::mat meas_design_mat_const,    // Internal matrix constraint
    const arma::mat meas_design_mat_mean,     // Internal matrix mean
    const arma::mat meas_design_mat_col_cov,  // Internal matrix covariance
    const arma::vec meas_design_mat_mean_alt, // Alternative prior formulation
    const arma::mat meas_design_mat_cov_alt,  // Alternative prior formulation
    const arma::mat meas_covar_mat_const,     // Covariate matrix mean
    const arma::mat meas_covar_mat_mean,      // Covariate matrix mean
    const arma::mat meas_covar_mat_col_cov,   // Covariate matrix covariance
    const arma::vec meas_covar_mat_mean_alt,  // Alternative prior formulation
    const arma::mat meas_covar_mat_cov_alt,   // Alternative prior formulation
    const uint &meas_cov_df,                  // Prior df dyn_cov
    const arma::mat &meas_cov_scale,          // Prior scale dyn_cov
    const bool uses_alt,                      // Alternative priors

    const arma::mat &y,
    bool exact = false,
    bool pred = false,
    bool disp_prog = true)
{
    // prior_sample currently uses covariate as a fixed input, which may
    // be appropriate for covariates like day of the week or time.
    // However, that means that it currently does not retrodictively sample
    // from it.

    // Initialize progress bar
    Progress p(n_iter, disp_prog);

    // ---------------------------------
    // Set-up
    // ---------------------------------

    // Initialize hyperparamters
    arma::vec hyperparameters(2);
    hyperparameters = Rcpp::as<arma::vec>(rprior());

    // Set data ---------
    arma::span sp_pred(0, n_time - 2);
    arma::span sp_out(1, n_time - 1);

    // Extract predictors
    arma::mat x(d_lat, n_time, arma::fill::zeros);
    arma::mat y_proxy(d_lat, n_time, arma::fill::zeros);
    arma::mat x_out = x.cols(sp_out);
    arma::mat x_pred = x.cols(sp_pred);
    arma::mat covariate_pred = covariate_dyn.cols(sp_pred);

    // Likelihood parameters
    arma::mat y_mean(d_obs, n_time, arma::fill::zeros);
    arma::mat y_cov_chol = identity(n_time);
    arma::mat x_mean(d_lat, n_time - 1, arma::fill::zeros);
    arma::mat x_cov_chol = identity(n_time - 1);

    // Measurement model parameters
    arma::mat meas_design_mat_cov_chol = chol(meas_design_mat_cov_alt, "lower");
    arma::mat meas_covar_mat_cov_chol = chol(meas_covar_mat_cov_alt, "lower");
    arma::mat meas_cov_scale_chol = chol(meas_cov_scale, "lower");

    mvn_covar_wrapper meas_model_wrapper(
        &x, &covariate_meas,
        &meas_design_mat_const, &meas_covar_mat_const,
        &meas_design_mat_mean_alt, &meas_covar_mat_mean_alt,
        &meas_design_mat_cov_chol, &meas_covar_mat_cov_chol);

    // Dynamic model parameters
    std::unique_ptr<gp_base> gp;
    if (exact)
    {
        gp = std::make_unique<imc_gp>();
        dyn_design_mat_mean.set_size(d_lat, n_time - 1);
    }
    else
    {
        gp = std::make_unique<hsgp_approx>(basis_fun_index,
                                           boundry_factor);
        dyn_design_mat_mean.set_size(d_lat, basis_fun_index.n_rows);
    }
    gp->set_hyperparameters(
        std::exp(hyperparameters[0]), std::exp(hyperparameters[1]));
    gp->update_predictor(x_pred);

    arma::mat dyn_covar_mat_cov_chol = chol(dyn_covar_mat_col_cov, "lower");
    arma::mat dyn_cov_scale_chol = chol(dyn_cov_scale, "lower");
    dyn_design_mat_mean.zeros();

    mn_covar_wrapper dyn_model_wrapper(
        gp->get_predictor_ptr(), &covariate_pred,
        &dyn_design_mat_mean, &dyn_covar_mat_mean,
        gp->get_cov_chol_ptr(), &dyn_covar_mat_cov_chol);

    // Initialize models
    mvn_iw_model_ meas_model = init_mvn_iw_model(
        y_proxy,
        y_mean,
        meas_model_wrapper,
        meas_cov_scale_chol,
        meas_cov_df);

    mn_iw_model_ dyn_model = init_mn_iw_model(
        y_proxy,
        x_mean,
        x_cov_chol,
        dyn_model_wrapper,
        dyn_cov_scale_chol,
        dyn_cov_df);

    // Lambda functions
    arma::vec par_vec;
    uint par_size;
    uint pos = 0;
    auto append = [&](const auto &v)
    {
        if (!v.n_elem)
            return;

        par_vec.subvec(pos, pos + v.n_elem - 1) = arma::vectorise(v);
        pos += v.n_elem;
    };

    // Initialize temparary and output matreces
    arma::mat samples;
    arma::mat covariate_ceof_temp;
    arma::mat y_pred(d_obs, n_time);
    arma::mat gp_sample(d_lat, n_time - 1);
    double log_lik;

    for (size_t k = 0; k < n_iter; k++)
    {
        // Increment progress bar
        if (Progress::check_abort())
        {
            return arma::mat(1, 1, arma::fill::zeros);
        }
        if (k % 10 == 0 && disp_prog)
        {
            p.increment(10);
        }
        // Sample hyperparameters
        hyperparameters = Rcpp::as<arma::vec>(rprior());
        gp->set_hyperparameters(
            std::exp(hyperparameters[0]), std::exp(hyperparameters[1]));
        // Sample measurement model
        meas_model.sample_prior();
        // Sample dynamic model
        dyn_model.sample_prior();
        // Sample latent variable
        run_sim_latent(
            x, covariate_dyn,
            *gp, dyn_model, dyn_model_wrapper,
            t0_mean, t0_cov,
            1);
        x_out = x.cols(sp_out);
        x_pred = x.cols(sp_pred);
        covariate_ceof_temp = dyn_model_wrapper.get_covar_param();
        // For the exact model - save the covariate sample and draw a new
        // GP model with a prior that conditions on x_pred.
        // So that covariate_ceof_temp is fixed from when x was sampled 
        // and the GP is marginalized when X was sampled and 
        // and sampled afterwards
        if (exact)
        {
            update_model_predictor(x_pred, *gp, dyn_model, dyn_model_wrapper);
            dyn_model.sample_prior();
        }

        // Sample derived properties
        meas_model_wrapper.combine_data();
        y_pred = meas_model.get_param() * meas_model_wrapper.combined_data;
        y_pred += chol(meas_model.get_cov(), "lower") *
                  arma::randn(d_obs, n_time, arma::distr_param(0.0, 1.0));
        gp_sample = dyn_model_wrapper.get_pred_param() *
                    (*gp->get_predictor_ptr());

        if (y.n_elem != 0)
        {
            log_lik = arma::sum(mat_logdnorm(y,
                                             arma::mat(meas_model.get_param() * meas_model_wrapper.combined_data),
                                             arma::chol(
                                                 meas_model.get_cov(), "lower")));
        }
        else
        {
            log_lik = arma::sum(mat_logdnorm(y_pred,
                                             arma::mat(
                                                 arma::mat(meas_model.get_param() * meas_model_wrapper.combined_data)),
                                             arma::chol(
                                                 meas_model.get_cov(), "lower")));
        }

        // Count number of parameters
        if (k == 0)
        {
            par_size = 0;
            if (pred)
                par_size += y_pred.n_elem;
            par_size += x.n_elem;
            par_size += gp_sample.n_elem;
            par_size += hyperparameters.n_elem;
            if (!exact)
                par_size += dyn_model_wrapper.get_pred_param().n_elem;
            par_size += covariate_ceof_temp.n_elem;
            par_size += dyn_model.get_cov().n_elem;
            par_size += meas_model_wrapper.get_pred_param().n_elem;
            par_size += meas_model_wrapper.get_covar_param().n_elem;
            par_size += meas_model.get_cov().n_elem;
            par_size += 1;
            par_vec.set_size(par_size);
            samples.set_size(n_iter, par_size);
        }

        pos = 0;
        if (pred)
        {
            append(y_pred);
        }
        append(x);
        append(gp_sample);
        append(arma::vec(arma::exp(hyperparameters)));
        if (!exact)
            append(dyn_model_wrapper.get_pred_param());
        append(covariate_ceof_temp);
        append(dyn_model.get_cov());
        append(meas_model_wrapper.get_pred_param());
        append(meas_model_wrapper.get_covar_param());
        append(meas_model.get_cov());

        append(arma::vec(1).fill(log_lik));

        // Store in samples matrix
        samples.row(k) = par_vec.t();

        // Collect all sampled variable in parameter vector
    }

    // Return samples matrix
    return samples;
}
