#ifndef HSGP_CLASS_H
#define HSGP_CLASS_H

// [[Rcpp::depends(RcppArmadillo)]]

// Hsgp approximation struct
struct hsgp_approx
{
    arma::mat indices;
    arma::vec boundry_factor;
    arma::mat sqrt_lambda;
    arma::vec spdf;
    arma::mat phi;
    double alpha;
    double rho;

    hsgp_approx(arma::mat indices_inp, arma::vec boundry_factor_inp);
    
    void update_hyperparameters(const double &alpha_new, const double &rho_new);
    void phi_transform(const arma::mat &x);
    arma::mat scaled_basis_functions(const arma::mat &x);
    arma::mat basis_functions(const arma::mat &x);
    arma::mat inv_scale();
    arma::mat scale();
};

#endif