// [[Rcpp::depends(RcppArmadillo)]]

#include <RcppArmadillo.h>
#include "niw_base_class.h"
#include "linear_algebra.h"

// Constructor
normal_inverse_wishart_base::normal_inverse_wishart_base(
    const arma::mat &des_int_mat_const,
    const arma::mat &des_covar_mat_const,
    uint cov_df_inp,
    const arma::mat &cov_scale_inp)
    : cov_df(cov_df_inp),
      cov_scale(cov_scale_inp)
{
    // Create model parameter and constraint matrix, by combining internal and
    // external paramters and constraints
    des_mat_const = join_rows(des_int_mat_const, des_covar_mat_const);

    // Fill in model dimensions
    d_outcome = des_mat_const.n_rows;
    d_predictor = des_mat_const.n_cols;

    // Resize parameter matrices
    des_int_mat.set_size(des_int_mat_const.n_rows,
                         des_int_mat_const.n_cols);
    des_covar_mat.set_size(des_covar_mat_const.n_rows,
                           des_covar_mat_const.n_cols);
    des_mat.set_size(d_outcome, d_predictor);
    cov.set_size(d_outcome, d_outcome);

    post_cov_scale.set_size(d_outcome, d_outcome);
}

//  Shared methods
void normal_inverse_wishart_base::update_data(const arma::mat &outcome_inp,
                                              const arma::mat &internal_inp,
                                              const arma::mat &covariate_inp)
{
    predictor = join_cols(internal_inp, covariate_inp);
    outcome = outcome_inp;
    n_time = outcome.n_cols;
}

void normal_inverse_wishart_base::fill_sample_into_des_mat(
    const arma::vec &sample)
{
    des_mat = vec2mat(sample, des_mat_const);
    des_int_mat = des_mat.cols(0, des_int_mat.n_cols - 1);
    if (des_int_mat.n_cols < des_mat.n_cols)
    {
        des_covar_mat = des_mat.cols(des_int_mat.n_cols, des_mat.n_cols - 1);
    }
    else
    {
        des_covar_mat = arma::mat(des_mat.n_rows, 0);
    }
}

void normal_inverse_wishart_base::sample_cov_prior()
{
    cov = arma::iwishrnd(cov_scale, cov_df);
}

void normal_inverse_wishart_base::sample_posterior_cov_cond_des_mat()
{
    arma::mat errors = outcome - des_mat * predictor;
    post_cov_scale = cov_scale + errors * errors.t();
    cov = arma::iwishrnd(
        post_cov_scale, n_time + cov_df);
}