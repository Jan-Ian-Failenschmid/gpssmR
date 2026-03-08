// Dependencies       ----
// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::depends(RcppProgress)]]
#include <RcppArmadillo.h>
#include <progress.hpp>
#include <progress_bar.hpp>
#include <cmath> // For M_PI
#include <variant>

#include "pdfs.h"
#include "linear_algebra.h"

#include "resampling.h"
#include "hsgp_class.h"
// #include "imc_iw_class.h"
#include "imc_gp_class.h"
#include "mh_class.h"
// #include "niw_base_class.h"
// #include "matniw_gp_class.h"
// #include "multniw_model_class.h"
#include "pgas.h"
#include "sim_latent.h"
#include "base_structs.h"
#include "derived_structs.h"

using namespace Rcpp;

// Export functions
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
    const arma::mat dyn_design_mat_mean,   // Internal matrix mean
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

  // ---------------------------------
  // Set-up
  // ---------------------------------

  // Initialize progress bar
  Progress p(n_total, disp_prog);

  // Initialize MH --------
  arma::vec hyperparameters(2);
  hyperparameters = Rcpp::as<arma::vec>(rprior());
  mh_kernel rw_mh(hyperparameters, dprior, n_warm_up, mh_adapt_start);

  // Wrap measurement covariate and predictor --------
  mvn_covar_wrapper mvn_wrapper(
      &x,
      &covariate_meas,
      meas_design_mat_const,
      meas_covar_mat_const,
      meas_design_mat_mean_alt,
      meas_covar_mat_mean_alt,
      meas_design_mat_cov_alt,
      meas_covar_mat_cov_alt);
  mvn_wrapper.set_data();

  // Define models --------
  // Dynamic model
  Rcpp::Rcout << "Test 1" << std::endl;
  std::unique_ptr<mn_conjugate_base> gp_model_ptr;
  imc_gp *gp_exact = nullptr;
  hsgp_approx *gp_approx = nullptr;
  if (exact)
  {
    auto gp = std::make_unique<imc_gp>();
    gp_exact = gp.get();

    gp_model_ptr = std::make_unique<mn_gp_mean_model_>(
        std::move(gp), identity(d_lat), 0);
  }
  else
  {
    auto hsgp = std::make_unique<hsgp_approx>(basis_fun_index,
                                              boundry_factor);
    gp_approx = hsgp.get();
    gp_model_ptr = std::make_unique<mn_hsgp_regression_model_>(
        std::move(hsgp), identity(d_lat), 0);
  }
  Rcpp::Rcout << "Test 2" << std::endl;
  auto dyn_covar = std::make_unique<mn_regression_model_>(
      dyn_covar_mat_mean, dyn_covar_mat_col_cov, identity(1), 1);

  auto dyn_gp_mn = std::make_unique<mn_mn_model_>(
      std::move(gp_model_ptr), std::move(dyn_covar));

  auto dyn_iw = std::make_unique<iw_model_>(dyn_cov_df, dyn_cov_scale);

  mn_iw_model_ dyn_model(std::move(dyn_gp_mn), std::move(dyn_iw));
  auto *mn_chain = dynamic_cast<mn_mn_model_ *>(dyn_model.mn.get());
  auto *gp_model = dynamic_cast<gp_base *>(mn_chain->mn1.get());

  // Measurement model
  auto meas_mvn = std::make_unique<mvn_regression_model_>(
      mvn_wrapper.get_mean(),
      mvn_wrapper.get_cov(),
      mvn_wrapper.get_const(),
      0);
  auto meas_iw = std::make_unique<iw_model_>(
      meas_cov_df,
      meas_cov_scale);

  mvn_iw_model_ meas_model(
      std::move(meas_mvn),
      std::move(meas_iw));
  Rcpp::Rcout << "Test 3" << std::endl;
  mvn_wrapper.set_param_ptr(&meas_model.mvn->param);

  arma::mat meas_data_mean(y.n_rows, y.n_cols, arma::fill::zeros);
  arma::mat dyn_data_mean(x.n_rows, x.n_cols - 1, arma::fill::zeros);

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

  auto set_gp_hyperpars = [&](const arma::vec &p)
  {
    gp_model->set_hyperparameters(std::exp(p[0]), std::exp(p[1]));
  };
  Rcpp::Rcout << "Test 4" << std::endl;
  // Set data ---------
  arma::span sp_pred(0, n_time - 2);
  arma::span sp_out(1, n_time - 1);

  // Extract predictors
  arma::mat x_out = x.cols(sp_out);
  arma::mat x_pred = x.cols(sp_pred);
  arma::mat covariate_pred = covariate_dyn.cols(sp_pred);

  Rcpp::Rcout << "Test 5" << std::endl;

  // Data
  Dataset meas_data{&y, {&mvn_wrapper.combined_data}};
  Dataset dyn_data{&x_out, {&x_pred, &covariate_pred}};

  // Hyperparameters need to be set before before data
  // Set initial parameter values -------
  Rcpp::Rcout << "Test 6" << std::endl;
  set_gp_hyperpars(hyperparameters);
  dyn_model.set_data(dyn_data, dyn_data_mean, identity(n_time - 1));
  dyn_model.calc_posterior_parameters();
  dyn_model.sample_posterior();

  meas_model.set_data(meas_data, meas_data_mean);
  for (size_t i = 0; i < 1000; i++)
  {
    // meas_model.sample_prior();
    meas_model.calc_posterior_parameters();
    meas_model.sample_posterior();
  }

  // Initialize latent vriable -------
  if (pg_rep > 0)
  {
    if (exact)
    {
      x = sim_latent(
          covariate_dyn, // Covariate data
          n_time,        // Number of time-points
          d_lat,         // State dimensions
          *gp_exact,
          t0_mean,
          t0_cov,
          mn_chain->mn2->get_param(), // State transition matrix
          dyn_model.iw->get_cov()     // Dynamic error matrix
      );
    }
    else
    {
      x = sim_latent(
          covariate_dyn, // Covariate data
          n_time,        // Number of time-points
          d_lat,         // State dimensions
          *gp_approx,
          t0_mean,
          t0_cov,
          mn_chain->mn1->get_param(),
          mn_chain->mn2->get_param(), // State transition matrix
          dyn_model.iw->get_cov()     // Dynamic error matrix
      );
    }
  }

  // Initialize temparary and output matreces
  arma::mat gp(d_lat, n_time - 1);
  arma::mat y_post_pred(d_obs, n_time);
  arma::mat samples;

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

    // Sample state smoothing -------
    for (size_t i = 0; i < pg_rep; i++)
    {
      if (exact)
      {
        x = pgas(
            y,
            covariate_dyn,
            covariate_meas,
            n_particles,
            n_time,
            d_lat,
            *gp_exact,
            x,
            t0_mean,
            t0_cov,
            mn_chain->mn2->get_param(),
            dyn_model.iw->get_cov(),
            mvn_wrapper.get_pred_param(),
            mvn_wrapper.get_covar_param(),
            meas_model.iw->get_cov());
      }
      else
      {
        x = pgas(
            y,
            covariate_dyn,
            covariate_meas,
            n_particles,
            n_time,
            d_lat,
            *gp_approx,
            x,
            t0_mean,
            t0_cov,
            mn_chain->mn1->get_param(),
            mn_chain->mn2->get_param(),
            dyn_model.iw->get_cov(),
            mvn_wrapper.get_pred_param(),
            mvn_wrapper.get_covar_param(),
            meas_model.iw->get_cov());
      }
    }

    // Reassing new data
    x_pred = x.cols(sp_pred);
    x_out = x.cols(sp_out);
    mvn_wrapper.set_data();

    // Sample measurement model parameters -------
    meas_model.calc_posterior_parameters();
    meas_model.sample_posterior();

    // Sample hyperparameters using MH-within-Gibbs -------
    // Marginalized MH-within-gibbs sampling for Hyperparameters
    // Start by calculating the log_marginal_likelihood of the last
    // set of hyperparameters for the new data
    set_gp_hyperpars(rw_mh.par);
    dyn_model.set_data(dyn_data, dyn_data_mean, identity(n_time - 1));
    dyn_model.calc_posterior_parameters();
    rw_mh.log_lik = dyn_model.log_marginal_likelihood();

    // MH - step
    rw_mh.advance_iter();
    for (size_t i = 0; i < mh_rep; i++)
    {
      rw_mh.make_proposal(); // Make proposal
      try
      {
        set_gp_hyperpars(rw_mh.prop_par);
        dyn_model.set_data(dyn_data, dyn_data_mean, identity(n_time - 1));
        dyn_model.calc_posterior_parameters();

        rw_mh.proposal_log_lik = dyn_model.log_marginal_likelihood();
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

    // Update hyperparameters
    set_gp_hyperpars(rw_mh.par);
    dyn_model.set_data(dyn_data, dyn_data_mean, identity(n_time - 1));

    // Sample dynamic model parameters -------
    dyn_model.calc_posterior_parameters();
    dyn_model.sample_posterior();

    // Sample dynamic model parameters -------
    gp = gp_model->get_gp_predictions();
    y_post_pred = meas_model.get_param() * mvn_wrapper.combined_data;
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
      par_size += gp.n_elem;
      par_size += rw_mh.par.n_elem;
      if (!exact)
        par_size += mn_chain->mn1->get_param().n_elem;
      par_size += mn_chain->mn2->get_param().n_elem;
      par_size += dyn_model.iw->get_cov().n_elem;
      par_size += mvn_wrapper.get_pred_param().n_elem;
      par_size += mvn_wrapper.get_covar_param().n_elem;
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
      append(gp);
      append(arma::vec(arma::exp(rw_mh.par)));
      if (!exact)
        append(mn_chain->mn1->get_param());
      append(mn_chain->mn2->get_param());
      append(dyn_model.iw->get_cov());
      append(mvn_wrapper.get_pred_param());
      append(mvn_wrapper.get_covar_param());
      append(meas_model.get_cov());

      // Store in samples matrix
      samples.row(k / n_thin) = par_vec.t();
    }
  }

  // Return samples matrix
  return samples;
};

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
    const arma::mat dyn_design_mat_mean,   // Internal matrix mean
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

  // Initialize hyperparamters
  arma::vec hyperparameters(2);
  hyperparameters = Rcpp::as<arma::vec>(rprior());

  // Set data ---------
  arma::span sp_pred(0, n_time - 2);
  arma::span sp_out(1, n_time - 1);

  // Extract predictors
  arma::mat x(d_lat, n_time);
  arma::mat x_out = x.cols(sp_out);
  arma::mat x_pred = x.cols(sp_pred);
  arma::mat covariate_pred = covariate_dyn.cols(sp_pred);

  // Data
  mvn_covar_wrapper mvn_wrapper(
      &x,
      &covariate_meas,
      meas_design_mat_const,
      meas_covar_mat_const,
      meas_design_mat_mean_alt,
      meas_covar_mat_mean_alt,
      meas_design_mat_cov_alt,
      meas_covar_mat_cov_alt);
  mvn_wrapper.set_data();

  Dataset meas_data{&y, {&mvn_wrapper.combined_data}};
  Dataset dyn_data{&x_out, {&x_pred, &covariate_pred}};

  // Initialize hsgp approximation
  hsgp_approx hsgp(basis_fun_index, boundry_factor);
  hsgp.update_hyperparameters(std::exp(hyperparameters[0]),
                              std::exp(hyperparameters[1]));

  std::unique_ptr<mn_conjugate_base> gp_model_ptr;
  imc_gp *gp_exact = nullptr;
  hsgp_approx *gp_approx = nullptr;
  if (exact)
  {
    auto gp = std::make_unique<imc_gp>();
    gp_exact = gp.get();

    gp_model_ptr = std::make_unique<mn_gp_mean_model_>(
        std::move(gp), identity(d_lat), 0);
  }
  else
  {
    auto hsgp = std::make_unique<hsgp_approx>(basis_fun_index,
                                              boundry_factor);
    gp_approx = hsgp.get();
    gp_model_ptr = std::make_unique<mn_hsgp_regression_model_>(
        std::move(hsgp), identity(d_lat), 0);
  }

  auto dyn_covar = std::make_unique<mn_regression_model_>(
      dyn_covar_mat_mean, dyn_covar_mat_col_cov, identity(0), 1);

  auto dyn_gp_mn = std::make_unique<mn_mn_model_>(
      std::move(gp_model_ptr), std::move(dyn_covar));

  auto dyn_iw = std::make_unique<iw_model_>(dyn_cov_df, dyn_cov_scale);

  mn_iw_model_ dyn_model(std::move(dyn_gp_mn), std::move(dyn_iw));
  auto *mn_chain = dynamic_cast<mn_mn_model_ *>(dyn_model.mn.get());
  auto *gp_model = dynamic_cast<gp_base *>(mn_chain->mn1.get());

  // Measurement model
  auto meas_mvn = std::make_unique<mvn_regression_model_>(
      mvn_wrapper.get_mean(),
      mvn_wrapper.get_cov(),
      mvn_wrapper.get_const(),
      0);
  auto meas_iw = std::make_unique<iw_model_>(
      meas_cov_df,
      meas_cov_scale);

  mvn_iw_model_ meas_model(
      std::move(meas_mvn),
      std::move(meas_iw));

  mvn_wrapper.set_param_ptr(&meas_model.mvn->param);

  arma::mat meas_data_mean(y.n_rows, y.n_cols, arma::fill::zeros);
  arma::mat dyn_data_mean(x.n_rows, x.n_cols - 1, arma::fill::zeros);

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

  auto set_gp_hyperpars = [&](const arma::vec &p)
  {
    gp_model->set_hyperparameters(std::exp(p[0]), std::exp(p[1]));
  };

  // Initialize temparary and output matreces
  arma::mat samples;
  arma::mat y_pred(d_obs, n_time);
  arma::mat gp(d_lat, n_time - 1);
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
    set_gp_hyperpars(hyperparameters);

    // Sample dynamic model
    // dyn_model.set_data(dyn_data, dyn_data_mean, identity(n_time - 1));

    // Sample measurement model
    // meas_model.set_data(meas_data, meas_data_mean);
    meas_model.sample_prior();

    // Sample latent variable
    if (exact)
    {
      dyn_model.iw->sample_prior();
      mn_chain->mn2->set_row_cov(dyn_model.iw->get_cov());
      mn_chain->mn1->set_row_cov(dyn_model.iw->get_cov());
      mn_chain->mn2->sample_prior();

      x = sim_latent(
          covariate_dyn, // Covariate data
          n_time,        // Number of time-points
          d_lat,         // State dimensions
          *gp_exact,
          t0_mean,
          t0_cov,
          mn_chain->mn2->get_param(), // State transition matrix
          dyn_model.iw->get_cov()     // Dynamic error matrix
      );

      x_out = x.cols(sp_out);
      x_pred = x.cols(sp_pred);
      dyn_model.set_data(dyn_data, dyn_data_mean, identity(n_time - 1));

      mn_chain->mn1->sample_prior();
    }
    else
    {
      dyn_model.sample_prior();
      x = sim_latent(
          covariate_dyn, // Covariate data
          n_time,        // Number of time-points
          d_lat,         // State dimensions
          *gp_approx,
          t0_mean,
          t0_cov,
          mn_chain->mn1->get_param(),
          mn_chain->mn2->get_param(), // State transition matrix
          dyn_model.iw->get_cov()     // Dynamic error matrix
      );

      x_out = x.cols(sp_out);
      x_pred = x.cols(sp_pred);
      dyn_model.set_data(dyn_data, dyn_data_mean, identity(n_time - 1));
    }

    gp = gp_model->get_gp_predictions();

    // Sample observed outcomes
    mvn_wrapper.set_data();
    y_pred = meas_model.get_param() * mvn_wrapper.combined_data;
    y_pred += chol(meas_model.get_cov(), "lower") *
              arma::randn(d_obs, n_time, arma::distr_param(0.0, 1.0));

    if (y.n_elem != 0)
    {
      log_lik = arma::sum(mat_logdnorm(y,
                                       arma::mat(meas_model.get_param() * mvn_wrapper.combined_data),
                                       arma::chol(
                                           meas_model.get_cov(), "lower")));
    }
    else
    {
      log_lik = arma::sum(mat_logdnorm(y_pred,
                                       arma::mat(
                                           arma::mat(meas_model.get_param() * mvn_wrapper.combined_data)),
                                       arma::chol(
                                           meas_model.get_cov(), "lower")));
    }

    // Count number of parameters
    if (k == 0)
    {
      par_size = 0;
      if (pred)
      {
        par_size += y_pred.n_elem;
      }
      par_size += x.n_elem;
      par_size += gp.n_elem;
      par_size += hyperparameters.n_elem;
      if (!exact)
        par_size += mn_chain->mn1->get_param().n_elem;
      par_size += mn_chain->mn2->get_param().n_elem;
      par_size += dyn_model.iw->get_cov().n_elem;
      par_size += mvn_wrapper.get_pred_param().n_elem;
      par_size += mvn_wrapper.get_covar_param().n_elem;
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
    append(gp);
    append(arma::vec(arma::exp(hyperparameters)));
    if (!exact)
      append(mn_chain->mn1->get_param());
    append(mn_chain->mn2->get_param());
    append(dyn_model.iw->get_cov());
    append(mvn_wrapper.get_pred_param());
    append(mvn_wrapper.get_covar_param());
    append(meas_model.get_cov());
    append(arma::vec(1).fill(log_lik));

    // Store in samples matrix
    samples.row(k) = par_vec.t();

    // Collect all sampled variable in parameter vector
  }

  // Return samples matrix
  return samples;
}
