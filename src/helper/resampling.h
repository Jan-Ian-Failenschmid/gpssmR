#ifndef RESAMPLING_H
#define RESAMPLING_H

#include <RcppArmadillo.h>

// Template function — header-only
template <typename RowType>
inline void softmax(RowType &&log_weights)
{
    log_weights -= arma::max(log_weights);
    log_weights = arma::exp(log_weights);
    log_weights /= arma::sum(log_weights);
}

inline arma::urowvec systematic_resampling(
    const arma::rowvec &w, // Vector of normalized resampling weights
    const arma::uword &N   // Size of resampled vector
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

    for (arma::uword i = 0; i < N; i++)
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
// Template function — header-only
template <typename T>
inline std::vector<T> resample_std_vector(const std::vector<T> &data,
                                         const arma::uvec &indices)
{
    std::vector<T> result;
    arma::uword n_resamples = indices.size();
    result.reserve(n_resamples);
    for (arma::uword i = 0; i < n_resamples; i++)
    {
        result.push_back(data[indices[i]]);
    }
    return result;
}

#endif