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

arma::urowvec systematic_resampling(
    const arma::rowvec &w, // Vector of normalized resampling weights
    const int &N           // Size of resampled vector
);

// Template function — header-only
template <typename T>
std::vector<T> resample_std_vector(const std::vector<T> &data,
                                   const arma::uvec &indices)
{
    std::vector<T> result;
    uint n_resamples = indices.size();
    result.reserve(n_resamples);
    for (size_t i = 0; i < n_resamples; i++)
    {
        result.push_back(data[indices[i]]);
    }
    return result;
}

#endif