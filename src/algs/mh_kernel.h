#ifndef MH_KERNEL_H
#define MH_KERNEL_H

#include <RcppArmadillo.h>

// MCMC kernel
struct mh_kernel
{
    arma::vec par;
    arma::vec prop_par;
    Rcpp::NumericVector R_par;
    Rcpp::NumericVector R_prop_par;

    Rcpp::Function dprior;

    double log_lik;
    double proposal_log_lik;
    double log_prior;
    double proposal_log_prior;

    double acceptance_rate;
    double acceptance_rate_sum;
    arma::uword acceptance_rate_n;

    arma::mat par_cov;
    arma::mat prop_cov;
    arma::mat prop_cov_chol;

    arma::uword iter;
    arma::uword d_par;
    double epsilon;

    // Adapting
    double delta;
    double average_error;
    double log_eps;
    double log_eps_bar;

    double init_epsilon;
    arma::uword warm_up;
    arma::uword adapt_start;
    arma::uword t0;
    double gamma;
    double kappa;

    arma::mat running_mean;

    mh_kernel(
        arma::vec par_inp,         // Parameter vector input
        Rcpp::Function dprior_inp, // Prior liklihood
        const arma::uword &warm_up_inp,
        const arma::uword &adapt_start_inp);

    void advance_iter();
    void reset_acceptance_rate();
    void finalize_acceptance_rate();
    void tune_proposal();
    void make_proposal();
    void mh_step();
};

#endif
