// [[Rcpp::depends(RcppArmadillo)]]
#include <RcppArmadillo.h>
#include "t_helper.h"

// Seeding
void set_r_seed(double seed)
{
    Rcpp::Environment base_env("package:base");
    Rcpp::Function set_seed_r = base_env["set.seed"];
    set_seed_r(std::floor(std::fabs(seed)));
};

// Compare double up to precision
bool compare_double(double x, double y, double precision)
{
    return std::abs(x - y) < precision;
};

// Compare matrix elemnt wise up to precision
bool compare_mat(const arma::mat &X, const arma::mat &Y, double precision)
{
    bool out = true;
    for (arma::uword i = 0; i < X.n_rows; i++)
    {
        for (arma::uword j = 0; j < X.n_cols; j++)
        {
            if (!compare_double(X(i, j), Y(i, j), precision))
                out = false;
        }
    }
    return out;
};