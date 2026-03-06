#ifndef GP_MODEL_BASE_CLASS_H
#define GP_MODEL_BASE_CLASS_H

// [[Rcpp::depends(RcppArmadillo)]]

struct gp_model_base
{
    virtual ~gp_model_base() = default;

    virtual void update_data(const arma::mat &outcome,
                             const arma::mat &predictor,
                             const arma::mat &covariate) = 0;
    virtual void update_hyperparameters(
        const double &alpha, const double &rho) = 0;
    virtual void calc_posterior_parameters() = 0;
    virtual void sample_prior() = 0;
    virtual void sample_joint_posterior() = 0;
    virtual arma::mat make_gp_predictions() = 0;

    // Marginal likelihood for MH
    virtual double log_marginal_likelihood() = 0;

    // Declare getters
    virtual const arma::mat &get_des_int_mat() const = 0;
    virtual const arma::mat &get_des_covar_mat() const = 0;
    virtual const arma::mat &get_cov() const = 0;
};

#endif