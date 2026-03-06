#ifndef MH_CLASS_H
#define MH_CLASS_H

// [[Rcpp::depends(RcppArmadillo)]]

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

    arma::mat par_cov;
    arma::mat prop_cov;
    arma::mat prop_cov_chol;

    uint iter;
    uint d_par;
    double epsilon;

    // Adapting
    double delta;
    double average_error;
    double log_eps;
    double log_eps_bar;

    double init_epsilon;
    uint warm_up;
    uint adapt_start;
    uint t0;
    double gamma;
    double kappa;

    arma::mat running_mean;

    mh_kernel(
        arma::vec par_inp,         // Parameter vector input
        Rcpp::Function dprior_inp, // Prior liklihood
        const uint &warm_up_inp,
        const uint &adapt_start_inp);

    void advance_iter();
    void tune_proposal();
    void make_proposal();
    void mh_step();
};

#endif