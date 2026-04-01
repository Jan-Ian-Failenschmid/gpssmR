// [[Rcpp::depends(RcppArmadillo)]]
#include <RcppArmadillo.h>
#include "base_structs.h"
#include "derived_structs.h"
#include "hsgp_struct.h"
#include "imc_gp_struct.h"
#include "sim_latent.h"
#include "pgas.h"
#include "main_helper.h"
#include "timer.h"

mn_iw_model_ init_mn_iw_model(
    arma::mat &Y,
    arma::mat &data_mean,
    arma::mat &data_cov,
    mn_covar_wrapper &model_wrapper,
    arma::mat &cov_scale_chol,
    double cov_df)
{
    auto iw = std::make_unique<iw_model_conjugate>(cov_df, &cov_scale_chol);

    auto mn = std::make_unique<mn_regression_model>(
        model_wrapper.get_mean_ptr(),
        model_wrapper.get_prior_cov_chol_ptr(),
        iw->get_cov_chol_ptr());

    mn->set_predictor(model_wrapper.get_data_ptr());
    model_wrapper.set_param_ptr(mn->get_coefficient_ptr());

    mn_iw_model_ model(std::move(mn), std::move(iw));

    model.set_outcome(&Y);
    model.set_likelihood_pars(&data_mean, &data_cov);
    // model.mn->calc_marginal_parameters();
    return model;
}

mvn_iw_model_ init_mvn_iw_model(
    arma::mat &Y,
    arma::mat &data_mean,
    mvn_covar_wrapper &model_wrapper,
    arma::mat &cov_scale_chol,
    double cov_df)
{
    auto iw = std::make_unique<iw_model_>(cov_df, &cov_scale_chol);

    auto mvn = std::make_unique<mvn_regression_model_>(
        model_wrapper.get_mean_ptr(),
        model_wrapper.get_prior_cov_chol_ptr(),
        model_wrapper.get_constraints_ptr());

    mvn->set_predictor(model_wrapper.get_data_ptr());
    model_wrapper.set_param_ptr(mvn->get_coefficient_ptr());

    mvn_iw_model_ model(std::move(mvn), std::move(iw));

    model.set_outcome(&Y);
    model.set_likelihood_pars(&data_mean);

    return model;
}

void update_model_hyperparameters(
    const arma::vec &hyperparameters,
    gp_base &gp,
    mn_iw_model_ &model,
    mn_covar_wrapper &wrapper)
{
    timer.tic("mh.gp_hyperpars");
    gp.set_hyperparameters(
        std::exp(hyperparameters[0]),
        std::exp(hyperparameters[1]));
    wrapper.combine_priors();
    model.mn->stabalize_col_cov_();
    model.calc_posterior_parameters();
    timer.toc("mh.gp_hyperpars");
    // model.mn->calc_marginal_parameters();
    // model.iw->calc_posterior_parameters();
}

void update_model_predictor(
    const arma::mat &raw_pred,
    gp_base &gp,
    mn_iw_model_ &model,
    mn_covar_wrapper &wrapper)
{
    gp.update_predictor(raw_pred);

    wrapper.combine_data();
    wrapper.combine_priors();

    model.mn->stabalize_col_cov_();
    model.calc_posterior_parameters();
    // model.mn->calc_marginal_parameters();
}

void run_sim_latent(
    arma::mat &x,
    const arma::mat &covariate_dyn,
    const gp_base &gp,
    const mn_iw_model_ &dyn_model,
    const mn_covar_wrapper &dyn_wrapper,
    const arma::vec &t0_mean,
    const arma::mat &t0_cov,
    arma::uword pg_rep)
{
    if (pg_rep == 0)
        return;

    arma::uword n_time = x.n_cols;
    arma::uword d_lat = x.n_rows;

    if (auto *m = dynamic_cast<const imc_gp *>(&gp))
    {
        x = sim_latent(
            covariate_dyn,
            n_time,
            d_lat,
            *m,
            t0_mean,
            t0_cov,
            dyn_wrapper.get_covar_param(),
            dyn_model.get_cov());
    }
    else if (auto *m = dynamic_cast<const hsgp_approx *>(&gp))
    {
        x = sim_latent(
            covariate_dyn,
            n_time,
            d_lat,
            *m,
            t0_mean,
            t0_cov,
            dyn_wrapper.get_pred_param(),
            dyn_wrapper.get_covar_param(),
            dyn_model.get_cov());
    }
    else
    {
        throw std::runtime_error("Unknown GP type in run_sim_latent()");
    }
}

void run_pgas(
    const arma::mat &y,
    arma::mat &x,
    const arma::mat &covariate_dyn,
    const arma::mat &covariate_meas,
    arma::uword n_particles,
    gp_base &gp,
    const mn_iw_model_ &dyn_model,
    const mn_covar_wrapper &dyn_wrapper,
    const mvn_iw_model_ &meas_model,
    const mvn_covar_wrapper &meas_wrapper,
    const arma::vec &t0_mean,
    const arma::mat &t0_cov,
    arma::uword pg_rep)
{
    arma::uword n_time = x.n_cols;
    arma::uword d_lat = x.n_rows;

    for (arma::uword i = 0; i < pg_rep; i++)
    {
        if (auto *m = dynamic_cast<imc_gp *>(&gp))
        {
            x = pgas(
                y,
                covariate_dyn,
                covariate_meas,
                n_particles,
                n_time,
                d_lat,
                *m,
                x,
                t0_mean,
                t0_cov,
                dyn_wrapper.get_covar_param(),
                dyn_model.get_cov(),
                meas_wrapper.get_pred_param(),
                meas_wrapper.get_covar_param(),
                meas_model.get_cov());
        }
        else if (auto *m = dynamic_cast<hsgp_approx *>(&gp))
        {
            x = pgas(
                y,
                covariate_dyn,
                covariate_meas,
                n_particles,
                n_time,
                d_lat,
                *m,
                x,
                t0_mean,
                t0_cov,
                dyn_wrapper.get_pred_param(),
                dyn_wrapper.get_covar_param(),
                dyn_model.get_cov(),
                meas_wrapper.get_pred_param(),
                meas_wrapper.get_covar_param(),
                meas_model.get_cov());
        }
        else
        {
            throw std::runtime_error("Unknown GP type in run_pgas()");
        }
    }
};
