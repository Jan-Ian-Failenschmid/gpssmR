// [[Rcpp::depends(RcppArmadillo)]]

#include <RcppArmadillo.h>
#include "linear_algebra.h"

// Matrix vector transfromations
arma::mat vec2mat(const arma::vec &vector, const arma::mat &mask)
{
    arma::mat out = mask;
    arma::uvec nan_ind = arma::find_nonfinite(mask);
    out.elem(nan_ind) = vector;
    return out;
}

arma::vec mat2vec(const arma::mat &matrix, const arma::mat &mask)
{
    return matrix.elem(arma::find_nonfinite(mask));
}

arma::uvec reorder_row2col(arma::uword n_rows, arma::uword n_cols)
{
    arma::uvec idx(n_rows * n_cols);

    for (arma::uword i = 0; i < n_rows; ++i)
    {
        for (arma::uword j = 0; j < n_cols; ++j)
        {
            arma::uword rowMajor = i * n_cols + j;
            arma::uword colMajor = j * n_rows + i;
            idx(colMajor) = rowMajor;
        }
    }
    return idx;
}

arma::mat diag_join(const arma::mat &A, const arma::mat &B)
{
    arma::mat result = arma::zeros<arma::mat>(
        A.n_rows + B.n_rows,
        A.n_cols + B.n_cols);

    result(arma::span(0, A.n_rows - 1), arma::span(0, A.n_cols - 1)) = A;
    result(arma::span(A.n_rows, A.n_rows + B.n_rows - 1),
           arma::span(A.n_cols, A.n_cols + B.n_cols - 1)) = B;

    return result;
}

// Expand grid
arma::mat expand_grid_2d(const arma::vec &a, const arma::vec &b)
{
    arma::mat grid(a.n_elem * b.n_elem, 2);

    grid.col(0) = arma::kron(arma::ones(b.n_elem), a);
    grid.col(1) = arma::kron(b, arma::ones(a.n_elem));
    return grid;
}

// Jitter matrix
void jitter_mat(arma::mat &K, double delta)
{
    K.diag() += delta;
}

// Make matrix symmetric
void make_symmetric(arma::mat &X)
{
    X = 0.5 * (X + X.t());
};

// Stabailized matrix inverse and cholesky
void stabalized_inv(arma::mat &cov, arma::mat &cov_inv, const arma::mat &cov_)
{
    try
    {
        if (cov_.is_diagmat())
        {
            arma::vec d = cov_.diag();
            arma::vec inv_d = 1.0 / d; // Invert diagmat

            inv_d.replace(arma::datum::inf,
                          std::numeric_limits<double>::max()); // Stabalize

            // Invert back
            cov = arma::diagmat(1.0 / inv_d);
            cov_inv = arma::diagmat(inv_d);
        }
        else
        {
            cov = cov_;
            cov_inv = arma::inv_sympd(cov);
        }
    }
    catch (const std::exception &e)
    {
        Rcpp::Rcout << "Exception caught in stabalized_inv_chol: "
                    << e.what() << std::endl;
    }
}

void fast_inv(arma::mat &cov_inv, const arma::mat &cov_)
{
    try
    {
        if (cov_.is_diagmat())
        {
            arma::vec d = cov_.diag();
            arma::vec inv_d = 1.0 / d; // Invert diagmat

            // Invert back
            cov_inv = arma::diagmat(inv_d);
        }
        else
        {
            cov_inv = arma::inv_sympd(cov_);
        }
    }
    catch (const std::exception &e)
    {
        Rcpp::Rcout << "Exception caught in fast_inv: "
                    << e.what() << std::endl;
    }
}

// Construct covariance matrix from cholesky pointer
void construct_cov(arma::mat &cov, const arma::mat &cov_chol) {
    cov = cov_chol * cov_chol.t();
};

arma::mat identity(arma::uword n)
{
    return arma::eye(n, n);
}

// Log determinants
double log_det_sympd_cust(const arma::mat &X)
{
    arma::mat L = arma::chol(X, "lower");
    return 2.0 * arma::sum(arma::log(L.diag()));
}

double log_det_chol(const arma::mat &L)
{
    return 2.0 * arma::sum(arma::log(L.diag()));
}

// Triangular solve
arma::mat chol_left_solve(const arma::mat &chol, const arma::mat &v)
{
    return arma::solve(arma::trimatl(chol), v, arma::solve_opts::fast);
};

// Cholesky updates
void chol_rank_one_update(
    arma::mat &L,
    double c,
    arma::vec x)
{
    uint n = x.n_elem;

    double ljj, ljj2, xj, xj2, ljj_new, gamma;
    double b = 1.0;
    for (size_t j = 0; j < n; j++) // Iterate over columns
    {
        ljj = L(j, j);
        ljj2 = ljj * ljj;
        xj = x(j);
        xj2 = xj * xj;

        ljj_new = std::sqrt(ljj2 + c / b * xj2); // New diagonal element
        L(j, j) = ljj_new;
        gamma = ljj2 * b + c * xj2;
        for (size_t i = (j + 1); i < n; i++) // Iterate over rows under diagonal
        {
            x(i) -= (xj / ljj) * L(i, j);
            L(i, j) = (ljj_new / ljj) * L(i, j) + (ljj_new * c * xj / gamma) * x(i);
        }
        b += c * xj2 / ljj2;
    }
}

arma::mat chol_of_sum(const arma::mat &L1, const arma::mat &L2)
{
    return arma::chol(L1 * L1.t() + L2 * L2.t(), "lower");
}

arma::mat chol_rank_n_update(
    arma::mat L,
    double c,
    arma::mat X)
{
    for (size_t i = 0; i < X.n_cols; i++)
    {
        chol_rank_one_update(L, c, X.col(i));
    }
    return L;
}

void chol_rank_n_update_ip(
    arma::mat &L,
    double c,
    const arma::mat &X)
{
    for (size_t i = 0; i < X.n_cols; i++)
    {
        chol_rank_one_update(L, c, X.col(i));
    }
}

void add_cholesky_lower(
    arma::mat &L,
    const arma::mat &lower_block,
    const arma::mat &diag_block)
{
    arma::mat diag_chol = arma::chol(diag_block, "lower");
    arma::uword L_rows = L.n_rows;
    arma::uword L_cols = L.n_cols;

    if (L_rows == 0)
    {
        L = diag_chol;
    }
    else
    {
        arma::mat lower_chol = chol_left_solve(L, lower_block.t()).t();
        chol_rank_n_update_ip(diag_chol, -1, lower_chol);
        L = diag_join(L, diag_chol);
        L.submat(L_rows, 0, L.n_rows - 1, L_cols - 1) = lower_chol;
    }
}
