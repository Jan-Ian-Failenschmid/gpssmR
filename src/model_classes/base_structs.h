#ifndef BASE_STRUCTS_H
#define BASE_STRUCTS_H

#include <RcppArmadillo.h>
#include "pdfs.h"
#include "linear_algebra.h"

struct Dataset
{
    const arma::mat *outcome;
    std::vector<const arma::mat *> predictors;
};

struct model_base
{
    // Pointers to data, read-only
    const arma::mat *outcome = nullptr;
    // const arma::mat *covariate;

    // Likelihood parameters
    arma::mat data_mean;
    arma::mat data_cov;
    arma::mat data_cov_chol;

    // Dimensionality
    arma::uword d_outcome = 0;
    arma::uword d_predictor = 0;
    arma::uword n_time = 0;

    // Log-likelihood
    double marginal_log_likelihood = 0.0;

    // Constructor / Destructor
    model_base() = default;
    virtual ~model_base() = default;

    // Virtual methods for polimorphism
    virtual void calc_posterior_parameters() = 0;
    virtual void sample_prior() = 0;
    virtual void sample_posterior() = 0;
    virtual double log_marginal_likelihood() = 0;

    // Setters
    // Set pointers to data
    virtual void set_data(const Dataset &data,
                          const arma::mat &data_mean_,
                          const arma::mat &data_cov_)
    {
        outcome = data.outcome;

        // covariate = covariate_ptr;

        data_mean = data_mean_;
        data_cov = data_cov_;
        data_cov_chol = arma::chol(data_cov, "lower");

        n_time = outcome->n_cols;
        d_outcome = outcome->n_rows;
        // d_predictor = predictor->n_rows;
    };
};

struct regression_base
{
    size_t predictor_idx;
    const arma::mat *predictor = nullptr;

    regression_base(size_t predictor_idx_)
        : predictor_idx(predictor_idx_) {}
    virtual ~regression_base() = default;
};

struct gp_base
{
    virtual ~gp_base() = default;

    virtual void set_hyperparameters(double alpha, double rho) = 0;
    virtual arma::mat get_gp_predictions() = 0;
};

// Base model for matrix normal likelihood with Inverse-Wishart prior
struct iw_model_ : public model_base
{
    arma::mat cov;

    // Prior parameters
    arma::uword v_prior;
    arma::mat cov_scale_prior;
    arma::mat cov_scale_prior_chol;

    // Posterior parameters
    arma::uword v_posterior;
    arma::mat cov_scale_posterior;
    arma::mat cov_scale_posterior_chol;

    // Helper
    arma::mat diff;
    arma::mat z;

    // Constructor / Destructor
    iw_model_(arma::uword v_prior_,
              const arma::mat &cov_scale_prior_)
        : v_prior(v_prior_),
          cov_scale_prior(cov_scale_prior_)
    {
        arma::uword d = cov_scale_prior.n_rows;

        cov_scale_prior_chol = arma::chol(cov_scale_prior, "lower");

        cov_scale_posterior.set_size(d, d);
        cov_scale_posterior_chol.set_size(d, d);
        cov.set_size(d, d);
    };

    void calc_posterior_parameters() override
    {
        diff = *outcome - data_mean;
        z = chol_left_solve(data_cov_chol, diff.t());

        v_posterior = v_prior + n_time;
        cov_scale_posterior = cov_scale_prior + z.t() * z;
        // cov_scale_posterior = cov_scale_prior + diff *
        //                                             arma::inv_sympd(data_cov) * diff.t();
        cov_scale_posterior_chol = arma::chol(cov_scale_posterior, "lower");
    };

    void sample_prior() override
    {
        arma::iwishrnd(cov, cov_scale_prior, v_prior);
    };

    void sample_posterior() override
    {
        arma::iwishrnd(cov, cov_scale_posterior, v_posterior);
    };

    double log_marginal_likelihood() override
    {
        marginal_log_likelihood = log_dmatrixt(
            data_cov_chol,
            cov_scale_prior_chol,
            cov_scale_posterior_chol,
            v_prior,
            v_posterior);
        return marginal_log_likelihood;
    };

    // Getters
    arma::mat get_cov()
    {
        return cov;
    };
};

struct mn_conjugate_base : public model_base
{
    arma::mat param;
    arma::mat param_prior;
    arma::mat param_posterior;

    arma::mat row_cov;
    arma::mat row_cov_chol;

    arma::mat col_cov_prior;
    arma::mat col_cov_prior_inv;
    arma::mat col_cov_prior_chol;

    arma::mat col_cov_posterior;
    arma::mat col_cov_posterior_chol;

    arma::mat data_cov_inv;

    mn_conjugate_base()
    {
        param_prior.set_size(0, 0);
        col_cov_prior.set_size(0, 0);
        row_cov.set_size(0, 0);
    };

    mn_conjugate_base(const arma::mat &param_prior_,
                      const arma::mat &col_cov_prior_,
                      const arma::mat &row_cov_)
    {
        set_param_prior(param_prior_);
        set_col_cov_prior(col_cov_prior_);
        set_row_cov(row_cov_);

        arma::uword d_cols = param_prior_.n_cols;
        arma::uword d_rows = param_prior_.n_rows;

        col_cov_posterior.set_size(d_cols, d_cols);
        col_cov_posterior_chol.set_size(d_cols, d_cols);
        param.set_size(d_rows, d_cols);
        param_posterior.set_size(d_rows, d_cols);
    }

    void sample_prior() override
    {
        rmatnorm(param, param_prior,
                 row_cov_chol, col_cov_prior_chol);
    }

    void sample_posterior() override
    {
        rmatnorm(param, param_posterior,
                 row_cov_chol, col_cov_posterior_chol);
    }

    void set_param_prior(const arma::mat &param_)
    {
        param_prior = param_;
    }

    virtual void set_row_cov(const arma::mat &row_cov_)
    {
        row_cov = row_cov_;
        row_cov_chol = chol(row_cov, "lower");
    }

    void set_col_cov_prior(const arma::mat &col_cov_)
    {
        stabalized_inv_chol(col_cov_prior,
                            col_cov_prior_chol,
                            col_cov_prior_inv,
                            col_cov_);
    }

    void set_data(const Dataset &data,
                  const arma::mat &data_mean_,
                  const arma::mat &data_cov_)
    {
        outcome = data.outcome;
        // predictor = predictor_ptr_;

        data_mean = data_mean_;
        stabalized_inv_chol(data_cov,
                            data_cov_chol,
                            data_cov_inv,
                            data_cov_);

        n_time = outcome->n_cols;
        d_outcome = outcome->n_rows;
        // d_predictor = predictor->n_rows;
    };

    // Getters
    virtual arma::mat get_marginal_mean() = 0;
    virtual arma::mat get_marginal_cov() = 0;
    
    arma::mat get_param()
    {
        return param;
    };
};

// Base model for matrix normal likelihood with matrix normal prior
struct mn_mean_model_ : public mn_conjugate_base
{
    mn_mean_model_(const arma::mat &param_prior_,
                   const arma::mat &col_cov_,
                   const arma::mat &row_cov_)
        : mn_conjugate_base(param_prior_, col_cov_, row_cov_) {};

    void calc_posterior_parameters() override
    {
        col_cov_posterior = arma::inv_sympd(data_cov_inv + col_cov_prior_inv);

        make_symmetric(col_cov_posterior);
        col_cov_posterior_chol = arma::chol(col_cov_posterior, "lower");

        param_posterior = (param_prior * col_cov_prior_inv +
                           (*outcome - data_mean) * data_cov_inv) *
                          col_cov_posterior;
    }

    double log_marginal_likelihood() override
    {
        arma::mat marginal_col_cov =
            data_cov + col_cov_prior;

        return logdmatnorm(
            *outcome,
            get_marginal_mean(),
            row_cov_chol,
            chol(get_marginal_cov(), "lower"));
    }

    void set_data(const Dataset &data,
                  const arma::mat &data_mean_,
                  const arma::mat &data_cov_)
    {
        mn_conjugate_base::set_data(data, data_mean_,
                                    data_cov_);

        param.resize(d_outcome, n_time);
        param_prior.resize(d_outcome, n_time);
        param_posterior.resize(d_outcome, n_time);
    };

    // Getters
    arma::mat get_marginal_mean() override
    {
        return data_mean + param_prior;
    };

    arma::mat get_marginal_cov() override
    {
        return data_cov + col_cov_prior;
    };
};

struct mn_regression_model_ : public mn_conjugate_base, public regression_base
{
    mn_regression_model_(const arma::mat &param_prior_,
                         const arma::mat &col_cov_,
                         const arma::mat &row_cov_,
                         size_t predictor_idx_)
        : mn_conjugate_base(param_prior_, col_cov_, row_cov_),
          regression_base(predictor_idx_) {};

    void calc_posterior_parameters() override
    {
        arma::mat Z = data_cov_inv * predictor->t();
        arma::mat sigma = *predictor * Z;
        arma::mat psi = (*outcome - data_mean) * Z;

        col_cov_posterior =
            arma::inv_sympd(sigma + col_cov_prior_inv);

        make_symmetric(col_cov_posterior);
        col_cov_posterior_chol = arma::chol(col_cov_posterior, "lower");

        param_posterior =
            (param_prior * col_cov_prior_inv + psi) * col_cov_posterior;
    }

    double log_marginal_likelihood() override
    {
        return logdmatnorm(
            *outcome, get_marginal_mean(), row_cov_chol,
            arma::chol(get_marginal_cov(), "lower"));
    }

    // Setters
    void set_data(const Dataset &data,
                  const arma::mat &data_mean_,
                  const arma::mat &data_cov_)
    {
        mn_conjugate_base::set_data(data, data_mean_,
                                    data_cov_);
        predictor = data.predictors[predictor_idx];
        d_predictor = predictor->n_rows;
    };

    // Getters
    arma::mat get_marginal_mean() override
    {
        return data_mean + param_prior * (*predictor);
    };

    arma::mat get_marginal_cov() override
    {
        return data_cov + predictor->t() * col_cov_prior * (*predictor);
    };
};

struct mvn_regression_model_ : public model_base, public regression_base
{
    arma::mat param; // Regression coefficients
    arma::vec sample_param;

    // Likelihood parameters
    arma::mat data_cov_inv;

    // Prior parameters
    arma::vec mean_prior;
    arma::mat cov_prior;
    arma::mat cov_prior_inv;
    arma::mat cov_prior_chol;

    // Posterior parameters
    arma::vec mean_posterior;
    arma::mat cov_posterior;
    arma::mat cov_posterior_chol;

    // Constraints
    arma::mat constraints;
    arma::uvec reorder_idx;
    arma::uvec free_idx;
    arma::uvec fix_idx;
    arma::uword n_free;
    arma::vec fix_coefs;

    // Helper
    arma::mat F_raw;
    arma::mat F_reorder;
    arma::mat F_fixed;
    arma::mat F_free;
    arma::vec z;
    arma::mat D;
    arma::vec d;

    mvn_regression_model_(const arma::mat &mean_prior_,
                          const arma::mat &cov_prior_,
                          const arma::mat &constraints_,
                          size_t predictor_idx_)
        : mean_prior(mean_prior_),
          cov_prior(cov_prior_),
          constraints(constraints_),
          regression_base(predictor_idx_)
    {
        param.set_size(constraints.n_rows, constraints.n_cols);

        cov_prior_inv = arma::inv_sympd(cov_prior);
        cov_prior_chol = chol(cov_prior, "lower");

        reorder_idx = reorder_row2col(
            constraints.n_rows, constraints.n_cols);
        free_idx = arma::find_nonfinite(constraints);
        fix_idx = arma::find_finite(constraints);
        n_free = free_idx.n_elem;
        fix_coefs = constraints.elem(fix_idx);

        sample_param.set_size(n_free);

        D.set_size(n_free, n_free);
        d.set_size(n_free);
    }

    void calc_posterior_parameters() override
    {
        // Sufficient statistics

        D.zeros();
        d.zeros();

        arma::mat identity = arma::eye(param.n_rows, param.n_rows);

        for (arma::uword t = 0; t < n_time; t++)
        {
            // Construct, reorder and split up F into fixed and free contribution
            F_raw = arma::kron(identity, predictor->col(t).t());
            F_reorder = F_raw.cols(reorder_idx);
            F_fixed = F_reorder.cols(fix_idx);
            F_free = F_reorder.cols(free_idx);

            // Remove fixed contribution from outcome
            z = outcome->col(t) - data_mean.col(t) - F_fixed * fix_coefs;

            D += F_free.t() * data_cov_inv * F_free;
            d += F_free.t() * data_cov_inv * z;
        }
        D += cov_prior_inv;
        d += cov_prior_inv * mean_prior;

        cov_posterior = arma::inv_sympd(D);
        cov_posterior_chol = arma::chol(cov_posterior, "lower");
        mean_posterior = cov_posterior * d;
    };

    void sample_prior() override
    {
        mat_rnorm(sample_param, mean_prior, cov_prior_chol);
        param = vec2mat(sample_param, constraints);
    };

    void sample_posterior() override
    {
        mat_rnorm(sample_param, mean_posterior, cov_posterior_chol);
        param = vec2mat(sample_param, constraints);
    };

    double log_marginal_likelihood() override
    {
        return 0;
    };

    void set_data(const Dataset &data,
                  const arma::mat &data_mean_,
                  const arma::mat &data_cov_)
    {
        outcome = data.outcome;
        predictor = data.predictors[predictor_idx];

        data_mean = data_mean_;
        stabalized_inv_chol(data_cov,
                            data_cov_chol,
                            data_cov_inv,
                            data_cov_);

        n_time = outcome->n_cols;
        d_outcome = outcome->n_rows;
        d_predictor = predictor->n_rows;
    };

    // Getters
    arma::mat get_param()
    {
        return param;
    };
};

#endif