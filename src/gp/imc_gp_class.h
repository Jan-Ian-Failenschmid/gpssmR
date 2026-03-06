#ifndef IMC_GP_CLASS_H
#define IMC_GP_CLASS_H

// [[Rcpp::depends(RcppArmadillo)]]

struct imc_gp
{
    uint dim_inp; // Input dimension
    uint dim_out; // Output dimension

    arma::mat train_dat; // (dim_inp × n_train)
    uint n_training;

    arma::mat test_dat; // (dim_inp × n_test)
    uint n_test;

    arma::mat outcome_dat; // (dim_out x n_train)
    uint n_outcome;

    arma::mat gp_val; // (dim_out x n_train)

    // Hyperparameters
    double alpha;    // signal variance
    double rho;      // lengthscale (shared across dims)
    arma::mat sigma; // output and error covariance (same for conjugacy)
    arma::mat sigma_chol;

    arma::mat train_mu; // training-set mean
    arma::mat test_mu;
    // arma::mat train_k;      // Training set kernel
    arma::mat test_train_k; // Kernel from training to test set
    // arma::mat test_k;       // Test set kernel
    arma::mat train_k_chol;
    arma::mat test_k_chol;

    arma::mat train_y_cov_chol;
    arma::mat test_y_cov_chol;

    arma::mat pred_mean;
    arma::mat pred_col_cov_chol;

    double train_norm_const;
    double log_lik;
    double marg_log_lik;

    double delta = 1e-8;

    void update_train_data(const arma::mat &training_data_inp,
                           const arma::mat &outcome_data_inp);
    void append_train_data(const arma::mat &training_data_inp,
                           const arma::mat &outcome_data_inp);
    void update_test_data(const arma::mat &test_data_inp);
    void append_test_data(const arma::mat &test_data_inp);
    void reset_test_data();
    void update_hyperparameters(const double &alpha_inp,
                                const double &rho_inp);
    void update_sigma(const arma::mat &sigma_inp);
    arma::mat kernel(const arma::mat &x1, const arma::mat &x2);
    arma::mat kernel(const arma::mat &x1);
    arma::mat mu(const arma::mat &x);
    double marginal_log_likelihood();
    double test_marginal_log_likelihood(const arma::mat &test_outcome);
    double prior_log_likelihood();
    void compute_predictive(const bool &marginal);
    arma::mat make_marginal_predictions();
    arma::mat make_test_predictions();
    arma::mat get_marginal_train_cov_chol();
    void set_train_y_cov(const arma::mat &y_cov_);
    void set_train_y_cov_I();
    void append_train_y_cov(const arma::mat &y_lower_block,
                            const arma::mat &y_lower_diag);
    void append_train_y_cov_I();
    void set_test_y_cov(const arma::mat &y_cov_);
    void set_test_y_cov_I();
    void append_test_y_cov(const arma::mat &y_lower_block,
                           const arma::mat &y_lower_diag);
    void append_test_y_cov_I();
    void reset_test_y_cov();
};

#endif