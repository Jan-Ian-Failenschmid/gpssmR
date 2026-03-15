#ifndef HSGP_STRUCT_H
#define HSGP_STRUCT_H

// [[Rcpp::depends(RcppArmadillo)]]

// Hsgp approximation struct
struct hsgp_approx : public gp_base
{
    arma::mat indices;
    arma::vec boundry_factor;
    arma::mat sqrt_lambda;
    arma::vec spdf;
    arma::mat scale_chol;
    arma::mat phi;
    double alpha;
    double rho;

    hsgp_approx(arma::mat indices_inp, arma::vec boundry_factor_inp);

    void update_hyperparameters(const double &alpha_new, const double &rho_new);
    void phi_transform(const arma::mat &x);
    void set_chol();
    arma::mat *get_cov_chol_ptr();
    arma::mat *get_predictor_ptr();
    arma::mat scaled_basis_functions(const arma::mat &x);
    arma::mat basis_functions(const arma::mat &x);
    arma::mat inv_scale();
    arma::mat scale();

    void set_hyperparameters(double alpha, double rho) override
    {
        update_hyperparameters(alpha, rho);
        set_chol();
    };

    void update_predictor(const arma::mat &predictor) override
    {
        phi_transform(predictor);
    };

    arma::mat get_gp_predictions() override
    {
        return arma::mat(0, 0);
    }
};

#endif