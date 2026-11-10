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

inline arma::urowvec mean_partition_order(
    const arma::rowvec& w // Vector of normalized resampling weights
)
{
    // Mean partition order algorithm (1) from the supplementary material of 
    // Conditional Particle Filters with Bridge Backward Sampling
    // Karppinen et al. (2024)
    // Adjusted for zero indexing 

    const arma::uword N = w.n_elem; // Number of weights
    const double p = arma::mean(w);

    arma::uword il = 0;
    arma::uword iu = N;

    arma::urowvec idx = arma::regspace<arma::urowvec>(0, N - 1);

    while (true)
    {
        // Find first weight > p
        while (il < std::min(iu, N))
        {
            if (w(idx(il)) > p)
            {
                break;
            }
            il++;
        }

        // Find last weight < p
        while (iu > il)
        {
            if (w(idx(iu - 1)) <= p)
            {
                break;
            }
            iu--;
        }

        if (il >= iu)
        {
            break;
        }

        idx.swap_cols(il, iu - 1);

        il++;
        iu--;
    }
    return idx;
}

inline arma::urowvec multinomial_resampling(
    const arma::rowvec& w, // Vector of normalized resampling weights
    const arma::uword& N   // Size of resampled vector
)
{
    // Initialize urowvec to hold indices (a)
    arma::urowvec idx(N, arma::fill::zeros);

    // Cummulative weights
    arma::rowvec cummulative_w = arma::cumsum(w);
    cummulative_w(cummulative_w.n_elem - 1) = 1.0;

    double u;
    for (arma::uword i = 0; i < N; i++)
    {
        u = arma::randu();

        for (arma::uword j = 0; j < w.n_elem; j++)
        {
            if (u < cummulative_w(j))
            {
                idx(i) = j;
                break;
            }
        }
        
    }
    return idx;
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
        while (q <= u && n < w.n_elem)
        {
            q += w(n);
            n++;
        }
        idx(i) = n - 1; // Store zero-based-indexing adjusted index
        u += 1.0 / N;
    }

    return idx;
}

inline arma::urowvec systematic_resampling(
    const arma::rowvec& w, // Vector of normalized resampling weights
    const arma::uword& N,   // Size of resampled vector
    double u // Sampling offset
)
{
    // Systematic resampling algorithm
    // Systematically resamples N index values based of of the importance weights

    // Initialize urowvec to hold indices (a)
    arma::urowvec idx(N, arma::fill::zeros);

    double q = 0.0; // Cumulative sum of weights
    arma::uword n = 0;      // Current index

    for (arma::uword i = 0; i < N; i++)
    {
        while (q <= u && n < w.n_elem)
        {
            q += w(n);
            n++;
        }
        idx(i) = n - 1; // Store zero-based-indexing adjusted index
        u += 1.0 / N;
    }

    return idx;
}

inline arma::urowvec conditional_systematic_resampling(
    const arma::rowvec& w, // Vector of normalized resampling weights
    const arma::uword aN // Ancestor of the n-th particle
)
{
    // Algorithm 6 from 
    // Conditional Particle Filters with Bridge Backward Sampling
    // Karppinen et al. (2024)
    // Adjusted for zero indexing 
    const arma::uword N = w.n_elem; // Number of ancestors

    // Calculate parameters for conditional systematic offset distribution
    // treating the aN-th ancestor as the first ancestor
    const double NW = N * w(aN);
    const arma::uword flNW = static_cast<arma::uword>(std::floor(NW));
    const double r = NW - flNW;

    // Sample conditional systematic offset and number of offspring of aN
    double u;
    arma::uword n_aN;

    if (arma::randu() < ((flNW + 1.0) * r) / NW)
    {
        // With probability (flNW + 1) * r) / NW
        // n_aN = flNW + 1 and sample u conditional on the higher 
        // offspring count from U(0, r)
        u = arma::randu() * r;
        n_aN = flNW + 1;
    }
    else {
        // else sample u from U(r, 1)
        u = r + arma::randu() * (1.0 - r);
        n_aN = flNW;
    }

    // Mean partition order of the weight vector
    arma::urowvec idx = mean_partition_order(w);

    // Find a position of aN in the mean partition
    const arma::uvec positions = arma::find(idx == aN);
    const arma::uword s = positions(0);

    // Cycle shift the mean partition so that aN is in the first positions
    // and conditional systematic offset calculation from above applies
    arma::urowvec idx_shifted(N);
    arma::rowvec w_reordered(N);

    for (arma::uword j = 0; j < N; j++)
    {
        idx_shifted(j) = idx((j + s) % N);
        w_reordered(j) = w(idx_shifted(j));
    }

    // Sample N systematic ancestor indices in the reordered system
    arma::urowvec a_reordered = systematic_resampling(w_reordered, N, u / N);

    // Restore original ancestor labels
    for (arma::uword j = 0; j < N; j++)
    {
        a_reordered(j) = idx_shifted(a_reordered(j));
    }

    // Uniformly select one of the NaN offspring of aN
    arma::uword Cbar = arma::randi<arma::uword>(
            arma::distr_param(0, n_aN - 1)
        );

    // Cyclically shift offspring positions so that A(N-1) = aN
    arma::urowvec a(N);

    for (arma::uword j = 0; j < N; j++)
    {
        a(j) = a_reordered((j + Cbar + 1) % N);
    }

    return a;
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