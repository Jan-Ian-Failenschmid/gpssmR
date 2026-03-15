// [[Rcpp::depends(RcppArmadillo)]]

#include <RcppArmadillo.h>
#include "mh_kernel.h"

// Constructor
mh_kernel::mh_kernel(
    arma::vec par_inp,         // Parameter vector input
    Rcpp::Function dprior_inp, // Prior liklihood
    const uint &warm_up_inp,
    const uint &adapt_start_inp) : par(par_inp), R_par(par_inp.begin(),
                                                       par_inp.end()),
                                   R_prop_par(), dprior(dprior_inp),
                                   prop_cov_chol(),
                                   log_lik(), t0(), warm_up(warm_up_inp),
                                   adapt_start(adapt_start_inp),
                                   proposal_log_lik(), prop_par(),
                                   acceptance_rate()
{
    d_par = par.size();
    iter = 0;
    log_lik = -std::numeric_limits<double>::max();

    epsilon = std::pow(2.38, 2) / par.size();
    par_cov = arma::eye(d_par, d_par) * 0.01;
    prop_cov = epsilon * par_cov;
    try
    {
        prop_cov_chol = chol(prop_cov, "lower");
    }
    catch (const std::exception &e)
    {
        Rcpp::Rcout << "Error in prop_cov_chol" << std::endl;
        prop_cov_chol = chol(prop_cov, "lower");
        std::cerr << e.what() << '\n';
    }

    log_prior = Rcpp::as<double>(dprior(R_par));

    delta = 0;
    average_error = 0;
    log_eps = std::log(epsilon);
    log_eps_bar = log_eps;

    init_epsilon = std::log(10 * epsilon);
    t0 = 10;
    gamma = 0.05;
    kappa = 0.75;

    running_mean = par;
}

void mh_kernel::advance_iter()
{
    iter += 1;
}

void mh_kernel::tune_proposal()
{
    if (iter > adapt_start && iter < warm_up)
    {
        delta = 0.234 - acceptance_rate;
        double weight = 1.0 / (iter + t0);
        average_error = (1.0 - weight) * average_error + weight * delta;
        log_eps = init_epsilon - (std::sqrt(iter) / gamma) * average_error;
        double smooth_weight = std::pow(iter, -kappa);
        log_eps_bar = smooth_weight * log_eps +
                      (1.0 - smooth_weight) * log_eps_bar;

        epsilon = std::exp(log_eps_bar);

        arma::vec delta = par - running_mean;
        running_mean += delta / iter;

        par_cov = ((iter - 2.0) / (iter - 1.0)) * par_cov +
                  (1.0 / iter) * delta * delta.t();

        // Add jitter and scale
        prop_cov = epsilon * (par_cov + 1e-6 * arma::eye(d_par, d_par));
        try
        {
            prop_cov_chol = chol(prop_cov, "lower");
        }
        catch (const std::exception &e)
        {
            Rcpp::Rcout << "Error in prop_cov_chol" << std::endl;
            prop_cov_chol = chol(prop_cov, "lower");
            std::cerr << e.what() << '\n';
        }
    }
}

void mh_kernel::make_proposal()
{
    prop_par = par + prop_cov_chol *
                         arma::randn(d_par, arma::distr_param(0.0, 1.0));
    R_prop_par = Rcpp::NumericVector(prop_par.begin(), prop_par.end());
}

void mh_kernel::mh_step()
{
    proposal_log_prior = Rcpp::as<double>(dprior(R_prop_par));
    acceptance_rate = proposal_log_lik - log_lik +
                      proposal_log_prior - log_prior;
    acceptance_rate = std::min(1.0, std::exp(acceptance_rate));
    if (arma::randu() < acceptance_rate)
    {
        par = prop_par;
        log_lik = proposal_log_lik;
        log_prior = proposal_log_prior;
    }
}