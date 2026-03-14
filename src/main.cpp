// Dependencies       ----
// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::depends(RcppProgress)]]
#include <RcppArmadillo.h>
#include <progress.hpp>
#include <progress_bar.hpp>
#include <cmath> // For M_PI
#include <variant>
#include <rcpptimer.h>
#include "pdfs.h"
#include "linear_algebra.h"

#include "resampling.h"
#include "hsgp_class.h"
#include "imc_iw_class.h"
#include "imc_gp_class.h"
#include "mh_class.h"
#include "niw_base_class.h"
#include "matniw_gp_class.h"
#include "multniw_model_class.h"
#include "pgas.h"
#include "sim_latent.h"
#include "base_structs.h"
#include "derived_structs.h"
#include "initializers.h"
#include "timer.h"

using namespace Rcpp;

// Export functions
// // [[Rcpp::export]]
// arma::mat gpssm_sample(

//     const uint &n_iter,     // Number of MCMC samples to be drawn
//     const uint &n_warm_up,  // Number of warm up samples to be drawn
//     const uint &n_thin,     //  Prethinning during sampling
//     const int &n_particles, // Number of particles

//     const uint &n_time,              // Number of time-points
//     const int &d_lat,                // Number of time-points
//     const int &d_obs,                // Number of time-points
//     arma::mat y,                     // Observation matrix y
//     arma::mat x,                     // Latent variable initialization
//     const arma::mat &covariate_dyn,  // Observation matrix y
//     const arma::mat &covariate_meas, // Observation matrix y

//     const arma::vec &t0_mean, // Latent mean at t0
//     const arma::mat &t0_cov,  // Latent mean at t0

//     const arma::mat &basis_fun_index, // Basis function index
//     const arma::vec &boundry_factor,  // Boundry factor
//     const Rcpp::Function &dprior,     // Hyperparameter log prior
//     const Rcpp::Function &rprior,     // Hyperparameter rng

//     const arma::mat dyn_design_mat_const,  // Internal matrix constraint
//     const arma::mat dyn_design_mat_mean,   // Internal matrix mean
//     const arma::mat dyn_covar_mat_const,   // Covariate matrix mean
//     const arma::mat dyn_covar_mat_mean,    // Covariate matrix mean
//     const arma::mat dyn_covar_mat_col_cov, // Covariate matrix covariance
//     const uint &dyn_cov_df,                // Prior df dyn_cov
//     const arma::mat &dyn_cov_scale,        // Prior scale dyn_cov

//     const arma::mat meas_design_mat_const,    // Internal matrix constraint
//     const arma::mat meas_design_mat_mean,     // Internal matrix mean
//     const arma::mat meas_design_mat_col_cov,  // Internal matrix covariance
//     const arma::vec meas_design_mat_mean_alt, // Alternative prior formulation
//     const arma::mat meas_design_mat_cov_alt,  // Alternative prior formulation
//     const arma::mat meas_covar_mat_const,     // Covariate matrix mean
//     const arma::mat meas_covar_mat_mean,      // Covariate matrix mean
//     const arma::mat meas_covar_mat_col_cov,   // Covariate matrix covariance
//     const arma::vec meas_covar_mat_mean_alt,  // Alternative prior formulation
//     const arma::mat meas_covar_mat_cov_alt,   // Alternative prior formulation
//     const uint &meas_cov_df,                  // Prior df dyn_cov
//     const arma::mat &meas_cov_scale,          // Prior scale dyn_cov
//     const bool uses_alt,                      // Alternative priors

//     const uint &mh_rep,
//     const uint &pg_rep,

//     const uint mh_adapt_start,
//     bool exact = false,
//     bool post_pred = false,
//     bool disp_prog = true)
// {
//     // Total iterations
//     uint n_total = (n_iter + n_warm_up) * n_thin;
//     Rcpp::Timer timer;
//     timer.reset();
//     timer.tic("setup");
//     // Initialize progress bar
//     Progress p(n_total, disp_prog);
//     Rcpp::Rcout << "Old" << std::endl;
//     // Initialize hyperparamters and MH kernel
//     arma::vec hyperparameters(2);
//     hyperparameters = Rcpp::as<arma::vec>(rprior());
//     mh_kernel rw_mh(hyperparameters, dprior, n_warm_up, mh_adapt_start);

//     // Initialize subsetting spans
//     arma::span sp_pred(0, n_time - 2);
//     arma::span sp_out(1, n_time - 1);

//     // Initialize mniw models
//     std::unique_ptr<gp_model_base> dynamic_model_ptr;
//     if (exact)
//     {
//         dynamic_model_ptr = std::make_unique<imc_iw>(
//             arma::exp(hyperparameters),
//             dyn_covar_mat_mean,
//             dyn_covar_mat_col_cov,
//             dyn_cov_df,
//             dyn_cov_scale);
//     }
//     else
//     {
//         dynamic_model_ptr = std::make_unique<matniw_gp_model>(
//             basis_fun_index,
//             boundry_factor,
//             arma::exp(hyperparameters),
//             dyn_design_mat_const, // Constraints
//             dyn_covar_mat_const,
//             dyn_design_mat_mean, // Means Needs to be zero atm
//             dyn_covar_mat_mean,  //
//             dyn_covar_mat_col_cov,
//             dyn_cov_df,   // Prior IW scale
//             dyn_cov_scale // Prior covariance scale
//         );
//     }

//     std::unique_ptr<normal_inverse_wishart_base> meas_model;
//     if (uses_alt)
//     {
//         meas_model = std::make_unique<multniw_model>(
//             meas_design_mat_const, // Constraints
//             meas_covar_mat_const,
//             meas_design_mat_mean_alt, // Means
//             meas_covar_mat_mean_alt,
//             meas_design_mat_cov_alt, // Column covariances
//             meas_covar_mat_cov_alt,
//             meas_cov_df,   // Prior IW scale
//             meas_cov_scale // Prior covariance scale
//         );
//     }
//     else
//     {
//         meas_model = std::make_unique<matniw_model>(
//             meas_design_mat_const, // Constraints
//             meas_covar_mat_const,
//             meas_design_mat_mean, // Means
//             meas_covar_mat_mean,
//             meas_design_mat_col_cov, // Column covariances
//             meas_covar_mat_col_cov,
//             meas_cov_df,   // Prior IW scale
//             meas_cov_scale // Prior covariance scale
//         );
//     }

//     // Find colums in y and latent start that contain missing values
//     // Logical vector marking which columns contain any NaN
//     arma::uvec nan_cols_y(n_time, arma::fill::zeros);
//     arma::uvec nan_cols_x(n_time, arma::fill::zeros);

//     for (arma::uword t = 0; t < n_time; ++t)
//     {
//         if (y.col(t).has_nan())
//         {
//             nan_cols_y[t] = 1;
//         }
//         if (x.col(t).has_nan())
//         {
//             nan_cols_x[t] = 1;
//         }
//     }
//     arma::uvec complete_case_x = arma::find(
//         nan_cols_x.tail(n_time - 1) + nan_cols_x.head(n_time - 1) == 0);
//     arma::uvec complete_case_y = arma::find(nan_cols_y + nan_cols_x == 0);

//     // Initialize model from reasonable values
//     arma::mat x_out = x.cols(sp_out);
//     arma::mat x_pred = x.cols(sp_pred);
//     arma::mat covariate_pred = covariate_dyn.cols(sp_pred);
//     dynamic_model_ptr->update_data(
//         x_out.cols(complete_case_x),
//         x_pred.cols(complete_case_x),
//         covariate_pred.cols(complete_case_x));
//     dynamic_model_ptr->calc_posterior_parameters();
//     dynamic_model_ptr->sample_joint_posterior();

//     if (uses_alt)
//     {
//         meas_model->sample_prior();
//         meas_model->update_data(
//             y.cols(complete_case_y),
//             x.cols(complete_case_y),
//             covariate_meas.cols(complete_case_y));
//         for (size_t i = 0; i < 1000; i++)
//         {
//             meas_model->calc_posterior_parameters();
//             meas_model->sample_posterior_des_mat_cond_cov();
//             meas_model->sample_posterior_cov_cond_des_mat();
//         }
//     }
//     else
//     {
//         meas_model->update_data(
//             y.cols(complete_case_y),
//             x.cols(complete_case_y),
//             covariate_meas.cols(complete_case_y));
//         meas_model->calc_posterior_parameters();
//         meas_model->sample_joint_posterior();
//     }

//     // dynamic_model.sample_prior();
//     // meas_model->sample_prior();
//     // Initialize latent variable
//       if (pg_rep > 0)
//       {
//         x = simulate_latent(
//             dynamic_model_ptr,
//             covariate_dyn,
//             n_time,
//             d_lat,
//             t0_mean,
//             t0_cov);
//       }

//     // Initialize models for missing observations
//     arma::mat data_mean = meas_model->des_int_mat * x +
//                           meas_model->des_covar_mat * covariate_meas;

//     std::vector<matniw_model> data_models;
//     data_models.reserve(arma::accu(nan_cols_y));

//     for (arma::uword t = 0; t < n_time; ++t)
//     {
//         if (!nan_cols_y[t])
//             continue;

//         data_models.emplace_back(
//             y.col(t),
//             arma::mat(d_obs, 0),
//             data_mean.col(t),
//             arma::mat(d_obs, 0),
//             arma::eye(1, 1),
//             arma::mat(0, 0),
//             meas_cov_df,
//             meas_cov_scale);

//         data_models.back().cov = meas_model->cov;
//         data_models.back().sample_prior_without_cov();
//         y.col(t) = data_models.back().des_int_mat;
//     }

//     // For convenience initialize a matrix that holds the tranformed GP values
//     arma::mat gp(d_lat, n_time - 1);
//     arma::mat y_post_pred(d_obs, n_time);

//     // Initialize samples matrix to hold MCMC samples
//     arma::mat samples;
//     arma::vec par_vec;
//     uint par_size;

//     uint pos = 0;
//     auto append = [&](const auto &v)
//     {
//         if (v.n_elem == 0)
//         {
//             return;
//         }
//         par_vec.subvec(pos, pos + v.n_elem - 1) = arma::vectorise(v);
//         pos += v.n_elem;
//     };
//     timer.toc("setup");
//     // Posterior sampling loop
//     for (size_t k = 0; k < n_total; k++)
//     {

//         // Increment progress bar
//         if (Progress::check_abort())
//         {
//             return arma::mat(1, 1, arma::fill::zeros);
//         }
//         if (k % 10 == 0 && disp_prog)
//         {
//             p.increment(10);
//         }
//         timer.tic("pgas");
//         // Conditional sampling of the states
//         // Possibility to sample multiple times in case of bad mixing
//         for (size_t i = 0; i < pg_rep; i++)
//         {
//           x = pgas(
//               y,
//               covariate_dyn,
//               covariate_meas,
//               n_particles,
//               n_time,
//               d_lat,
//               *dynamic_model_ptr, // Dereference unique_ptr
//               *meas_model,        // Dereference unique_ptr
//               x,                  // Previous latent sample
//               t0_mean,
//               t0_cov);
//         }
//         timer.toc("pgas");
//         // Conditional sampling of the measurement model
//         meas_model->update_data(y, x, covariate_meas);
//         timer.tic("meas_model_posterior");
//         meas_model->calc_posterior_parameters();

//         // This ordering is important because the posterior parameters are
//         // calculated using the previous cov
//         meas_model->sample_posterior_des_mat_cond_cov();
//         meas_model->sample_posterior_cov_cond_des_mat();
//         timer.toc("meas_model_posterior");

//         // Marginalized MH-within-gibbs sampling for Hyperparameters
//         // Start by calculating the log_marginal_likelihood of the last
//         // set of hyperparameters for the new data
//         timer.tic("mh");
//         dynamic_model_ptr->update_data(
//             x.cols(sp_out),
//             x.cols(sp_pred),
//             covariate_dyn.cols(sp_pred));
//         dynamic_model_ptr->update_hyperparameters(
//             std::exp(rw_mh.par[0]), std::exp(rw_mh.par[1]));
//         dynamic_model_ptr->calc_posterior_parameters();

//         rw_mh.log_lik = dynamic_model_ptr->log_marginal_likelihood();

//         // MH - step
//         rw_mh.advance_iter();
//         for (size_t i = 0; i < mh_rep; i++)
//         {
//           rw_mh.make_proposal(); // Make proposal
//           try
//           {
//             timer.tic("mh.set_hyperpars");
//             dynamic_model_ptr->update_hyperparameters(
//                 std::exp(rw_mh.prop_par[0]), std::exp(rw_mh.prop_par[1]));
//             timer.toc("mh.set_hyperpars");
//             timer.tic("mh.calc_pars");
//             dynamic_model_ptr->calc_posterior_parameters();
//             timer.toc("mh.calc_pars");
//             timer.tic("mh.calc_ll");
//             rw_mh.proposal_log_lik = dynamic_model_ptr->log_marginal_likelihood();
//             timer.toc("mh.calc_ll");
//           }
//           catch (const std::exception &e)
//           {
//             Rcpp::Rcout << "Warning: The current Metropolis-Hastings proposal is about to be rejectected, because a parameter is outside of the valid range." << std::endl;
//             Rcpp::Rcout << "Exception caught in MH proposal: " << e.what() << std::endl;
//             Rcpp::Rcout << std::exp(rw_mh.prop_par[0]) << std::endl;
//             Rcpp::Rcout << std::exp(rw_mh.prop_par[1]) << std::endl;
//             rw_mh.proposal_log_lik = -std::numeric_limits<double>::max();
//           }

//           rw_mh.mh_step(); // Accept or reject proposal
//         }
//         rw_mh.tune_proposal(); // Tune proposal based on acceptance ratio
//         timer.toc("mh");
//         // Update approximation with new hyperparameters after MH-step
//         timer.tic("dyn_model_posterior");
//         dynamic_model_ptr->update_hyperparameters(
//             std::exp(rw_mh.par[0]), std::exp(rw_mh.par[1]));

//         // Conditional sampling of the dynamic model
//         // Data does not need to be update because it is already update during
//         // the marginal likelihood calculation
//         dynamic_model_ptr->calc_posterior_parameters();
//         dynamic_model_ptr->sample_joint_posterior();
//         timer.toc("dyn_model_posterior");
//         // Sample generated quantities
//         gp = dynamic_model_ptr->make_gp_predictions();
//         y_post_pred = meas_model->des_int_mat * x +
//                       meas_model->des_covar_mat * covariate_meas;
//         y_post_pred += chol(meas_model->cov, "lower") *
//                        arma::randn(d_obs, n_time, arma::distr_param(0.0, 1.0));

//         // Count number of parameters
//         if (k == 0)
//         {
//             par_size = 0;
//             if (post_pred)
//                 par_size += y_post_pred.n_elem;
//             par_size += x.n_elem;
//             par_size += gp.n_elem;
//             par_size += rw_mh.par.n_elem;
//             if (!exact)
//                 par_size += dynamic_model_ptr->get_des_int_mat().n_elem;
//             par_size += dynamic_model_ptr->get_des_covar_mat().n_elem;
//             par_size += dynamic_model_ptr->get_cov().n_elem;
//             par_size += meas_model->des_int_mat.n_elem;
//             par_size += meas_model->des_covar_mat.n_elem;
//             par_size += meas_model->cov.n_elem;

//             par_vec.set_size(par_size);
//             samples.set_size(n_total / n_thin, par_size);
//         }

//         // Sample new set of missing data points
//         data_mean = meas_model->des_int_mat * x +
//                     meas_model->des_covar_mat * covariate_meas;

//         arma::uword model_idx = 0;
//         for (arma::uword t = 0; t < n_time; ++t)
//         {
//             if (!nan_cols_y[t])
//                 continue;
//             auto &dm = data_models[model_idx++]; // get the correct model
//             dm.update_mean(data_mean.col(t), arma::mat(d_obs, 0));
//             dm.cov = meas_model->cov;
//             dm.sample_prior_without_cov();
//             y.col(t) = dm.des_int_mat;
//         }
//         // Fill parameter vector
//         if (k % n_thin == 0)
//         {
//             pos = 0;
//             if (post_pred)
//             {
//                 append(y_post_pred);
//             }
//             append(x);
//             append(gp);
//             append(arma::vec(arma::exp(rw_mh.par)));
//             if (!exact)
//                 append(dynamic_model_ptr->get_des_int_mat());
//             append(dynamic_model_ptr->get_des_covar_mat());
//             append(dynamic_model_ptr->get_cov());
//             append(meas_model->des_int_mat);
//             append(meas_model->des_covar_mat);
//             append(meas_model->cov);

//             // Store in samples matrix
//             samples.row(k / n_thin) = par_vec.t();
//         }
//     }

//     // Return samples matrix
//     return samples;
// };

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
    Rcpp::Rcout << "New" << std::endl;

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

        x_out = x.cols(sp_out);
        x_pred = x.cols(sp_pred);
        update_model_predictor(x_pred, *gp, dyn_model, dyn_model_wrapper);
        meas_model_wrapper.combine_data();

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
            rw_mh.make_proposal(); // Make proposal
            try
            {
                timer.tic("mh.set_hyperpars1");
                update_model_hyperparameters(rw_mh.prop_par, *gp, dyn_model,
                                             dyn_model_wrapper);
                timer.toc("mh.set_hyperpars2");
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

            rw_mh.mh_step(); // Accept or reject proposal
        }
        rw_mh.tune_proposal(); // Tune proposal based on acceptance ratio

        // // Update hyperparameters
        update_model_hyperparameters(rw_mh.par, *gp, dyn_model,
                                     dyn_model_wrapper);
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
