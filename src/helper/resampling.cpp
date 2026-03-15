// [[Rcpp::depends(RcppArmadillo)]]
#include <RcppArmadillo.h>
#include "resampling.h"

// Systematir resampling helper function used in PGAS to sample from
// Importance weights
arma::urowvec systematic_resampling(
    const arma::rowvec &w, // Vector of normalized resampling weights
    const int &N           // Size of resampled vector
)
{
    // Systematic resampling algorithm
    // Systematically resamples N index values based of of the importance weights

    // Draw random double
    double u = arma::randu() / N;

    // Initialize urowvec to hold indices (a)
    arma::urowvec idx(N, arma::fill::zeros);

    double q = 0.0; // Cumulative sum of weights
    int n = 0;      // Current index

    for (size_t i = 0; i < N; i++)
    {
        while (q < u)
        {
            q += w(n);
            n++;
        }
        idx(i) = n - 1; // Store zero-based-indexing adjusted index
        u += 1.0 / N;
    }

    return idx;
}