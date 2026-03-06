#ifndef NIW_BASE_H
#define NIW_BASE_H

// [[Rcpp::depends(RcppArmadillo)]]
#include <RcppArmadillo.h>

struct normal_inverse_wishart_base
{
  // Design matrices
  arma::mat des_int_mat;
  arma::mat des_covar_mat;
  arma::mat des_mat;
  arma::mat des_mat_const;

  // Covariance
  arma::mat cov;
  uint cov_df;
  arma::mat cov_scale;

  // Data
  arma::mat outcome;
  arma::mat predictor;

  uint d_outcome;
  uint d_predictor;
  uint n_time;

  arma::mat post_cov_scale;

  // Likelihood bookkeeping
  double log_lik;
  double log_det_Lambda_nod;
  double log_det_Lambda_n;
  double log_det_prior_cov_scale;
  double log_det_post_cov_scale;

  normal_inverse_wishart_base(
      const arma::mat &des_int_mat_const,
      const arma::mat &des_covar_mat_const,
      uint cov_df_inp,
      const arma::mat &cov_scale_inp);

  virtual ~normal_inverse_wishart_base() = default; // Auto destructor

  void update_data(const arma::mat &outcome_inp,
                   const arma::mat &internal_inp,
                   const arma::mat &covariate_inp);
  void fill_sample_into_des_mat(const arma::vec &sample);
  void sample_cov_prior();
  void sample_posterior_cov_cond_des_mat();

  // Declare virtual member functions
  virtual void sample_prior() = 0;
  virtual void calc_posterior_parameters() = 0;
  virtual void sample_joint_posterior() = 0;
  virtual void sample_posterior_des_mat_cond_cov() = 0;

};

#endif