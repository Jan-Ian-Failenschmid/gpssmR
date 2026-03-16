#ifndef BASE_STRUCTS_H
#define BASE_STRUCTS_H

#include <RcppArmadillo.h>
#include "pdfs.h"
#include "linear_algebra.h"

struct model_base
{
    // Data pointers (non owning)
    const arma::mat *outcome = nullptr;

    arma::mat *data_mean = nullptr;
    arma::mat data_cov;
    arma::mat *data_cov_chol = nullptr;

    arma::uword d_outcome = 0;
    arma::uword d_predictor = 0;
    arma::uword n_time = 0;

    double marginal_log_likelihood = 0.0;

    model_base() = default;
    virtual ~model_base() = default;

    void set_outcome(const arma::mat *outcome_)
    {
        outcome = outcome_;
        n_time = outcome->n_cols;
    }

    virtual void calc_posterior_parameters() = 0;
    virtual void sample_prior() = 0;
    virtual void sample_posterior() = 0;
    virtual double log_marginal_likelihood() = 0;
};

struct regression_base
{
    const arma::mat *predictor = nullptr;

    arma::mat coefficient;

    regression_base() = default;
    virtual ~regression_base() = default;

    // Setters
    void set_predictor(const arma::mat *predictor_)
    {
        predictor = predictor_;
    }

    // Getters
    arma::mat get_coefficient() const
    {
        return coefficient;
    }

    arma::mat *get_coefficient_ptr()
    {
        return &coefficient;
    }
};

struct gp_base
{
    virtual ~gp_base() = default;

    virtual void set_hyperparameters(double alpha, double rho) = 0;
    virtual arma::mat get_gp_predictions() = 0; // Remove eventually
    virtual void update_predictor(const arma::mat &predictor) = 0;
    virtual arma::mat *get_cov_chol_ptr() = 0;
    virtual arma::mat *get_predictor_ptr() = 0;
};

// Base model for matrix normal likelihood with Inverse-Wishart prior
struct iw_model_ : public model_base
{
    arma::mat cov;
    arma::mat chol_cov;

    arma::uword v_prior;
    arma::mat cov_scale_prior;
    arma::mat *cov_scale_prior_chol = nullptr;

    arma::uword v_posterior;
    arma::mat cov_scale_posterior;
    arma::mat cov_scale_posterior_chol;

    arma::mat diff;
    arma::mat z;

    iw_model_(arma::uword v_prior_,
              arma::mat *cov_scale_prior_chol_)
        : v_prior(v_prior_),
          cov_scale_prior_chol(cov_scale_prior_chol_)
    {
        construct_cov(cov_scale_prior, *cov_scale_prior_chol);

        arma::uword d = cov_scale_prior.n_rows;

        cov_scale_posterior.set_size(d, d);
        cov_scale_posterior_chol.set_size(d, d);
        cov.set_size(d, d);
    }

    void calc_posterior_parameters() override
    {
        diff = *outcome - *data_mean;

        z = chol_left_solve(*data_cov_chol, diff.t());

        v_posterior = v_prior + n_time;

        cov_scale_posterior = cov_scale_prior + z.t() * z;
        // cov_scale_posterior = cov_scale_prior + diff *
        //                                             arma::inv_sympd(data_cov) * diff.t();
        cov_scale_posterior_chol = arma::chol(cov_scale_posterior, "lower");
    }

    void sample_prior() override
    {
        arma::iwishrnd(cov, cov_scale_prior, v_prior);
        chol_cov = arma::chol(cov, "lower");
    }

    void sample_posterior() override
    {
        arma::iwishrnd(cov, cov_scale_posterior, v_posterior);
        chol_cov = arma::chol(cov, "lower");
    }

    double log_marginal_likelihood() override
    {
        marginal_log_likelihood = log_dmatrixt(
            *data_cov_chol,
            *cov_scale_prior_chol,
            cov_scale_posterior_chol,
            v_prior,
            v_posterior);

        return marginal_log_likelihood;
    }

    void set_likelihood_pars(arma::mat *data_mean_, arma::mat *cov_chol_)
    {
        data_mean = data_mean_;
        data_cov_chol = cov_chol_;
    };

    // Getters
    arma::mat get_cov() const
    {
        return cov;
    }

    arma::mat *get_cov_ptr()
    {
        return &cov;
    }

    arma::mat get_cov_chol()
    {
        return chol_cov;
    }

    arma::mat *get_cov_chol_ptr()
    {
        return &chol_cov;
    }
};

struct mn_regression_model : public model_base, public regression_base
{
    arma::mat *coefficient_prior = nullptr;
    arma::mat coefficient_posterior;

    arma::mat row_cov;
    arma::mat *row_cov_chol = nullptr;

    arma::mat col_cov_prior;
    arma::mat col_cov_prior_inv;
    arma::mat *col_cov_prior_chol = nullptr;

    arma::mat col_cov_posterior;
    arma::mat col_cov_posterior_chol;

    arma::mat data_cov_inv;

    arma::mat marginal_mean;
    arma::mat marginal_cov;
    arma::mat marginal_cov_chol;

    mn_regression_model(arma::mat *coefficient_prior_,
                        arma::mat *col_cov_prior_chol_,
                        arma::mat *row_cov_chol_)
        : coefficient_prior(coefficient_prior_),
          row_cov_chol(row_cov_chol_),
          col_cov_prior_chol(col_cov_prior_chol_)
    {
        stabalize_col_cov_();

        arma::uword d_rows = coefficient_prior->n_rows;
        arma::uword d_cols = coefficient_prior->n_cols;

        col_cov_posterior.set_size(d_cols, d_cols);
        col_cov_posterior_chol.set_size(d_cols, d_cols);

        coefficient.set_size(d_rows, d_cols);
        coefficient_posterior.set_size(d_rows, d_cols);
    }

    void calc_posterior_parameters() override
    {
        arma::mat Z = data_cov_inv * predictor->t();
        arma::mat sigma = (*predictor) * Z;
        arma::mat psi = (*outcome - *data_mean) * Z;

        col_cov_posterior = arma::inv_sympd(sigma + col_cov_prior_inv);
        make_symmetric(col_cov_posterior);

        col_cov_posterior_chol = arma::chol(col_cov_posterior, "lower");
        coefficient_posterior = ((*coefficient_prior) *
                                     col_cov_prior_inv +
                                 psi) *
                                col_cov_posterior;
    }

    void calc_marginal_mean()
    {
        marginal_mean = *data_mean + (*coefficient_prior) * (*predictor);
    }

    void calc_marginal_cov()
    {
        marginal_cov = data_cov + predictor->t() * col_cov_prior * (*predictor);
        marginal_cov_chol = arma::chol(marginal_cov, "lower");
        // marginal_cov_chol = chol_rank_n_update(*data_cov_chol, 1, 
        //                                        predictor->t() * *col_cov_prior_chol);
    }

    void calc_marginal_parameters()
    {
        calc_marginal_mean();
        calc_marginal_cov();
    }

    double log_marginal_likelihood() override
    {
        return logdmatnorm(
            *outcome,
            marginal_mean,
            *row_cov_chol,
            marginal_cov_chol);
    }

    void sample_prior() override
    {
        rmatnorm(coefficient,
                 *coefficient_prior,
                 *row_cov_chol,
                 *col_cov_prior_chol);
    }

    void sample_posterior() override
    {
        rmatnorm(coefficient,
                 coefficient_posterior,
                 *row_cov_chol,
                 col_cov_posterior_chol);
    }

    // Setters
    void set_coef_prior(arma::mat *coefficient_prior_)
    {
        coefficient_prior = coefficient_prior_;
    }

    void set_row_cov(arma::mat *row_cov_chol_)
    {
        row_cov_chol = row_cov_chol_;
    }

    void set_col_cov_prior(arma::mat *col_cov_chol_)
    {
        col_cov_prior_chol = col_cov_chol_;
    }

    void stabalize_col_cov_()
    {
        // construct_cov(col_cov_prior, *col_cov_prior_chol);
        // If I get problems later on I need to fix this
        col_cov_prior = (*col_cov_prior_chol) * col_cov_prior_chol->t();
        fast_inv(col_cov_prior_inv, col_cov_prior);
        // arma::vec inv_spdf = 1.0 / col_cov_prior.diag();
        // col_cov_prior_inv = arma::diagmat(inv_spdf);
        // stabalized_inv(col_cov_prior,
        //                col_cov_prior_inv,
        //                (*col_cov_prior_chol) * col_cov_prior_chol->t());
    }

    void set_likelihood_pars(arma::mat *data_mean_, arma::mat *cov_chol_)
    {
        data_mean = data_mean_;
        data_cov_chol = cov_chol_;
        data_cov = (*data_cov_chol) * data_cov_chol->t();
        fast_inv(data_cov_inv, data_cov);
        // data_cov_inv = arma::diagmat(1.0 / data_cov.diag());
            // stabalized_inv(data_cov,
            //                data_cov_inv,
            //                (*data_cov_chol) * data_cov_chol->t());
    };

    // Getters
    arma::mat get_marginal_mean()
    {
        return marginal_mean;
    }

    arma::mat get_marginal_cov()
    {
        return marginal_cov;
    }

    arma::mat *get_marginal_mean_ptr()
    {
        return &marginal_mean;
    }

    arma::mat *get_marginal_cov_ptr()
    {
        return &marginal_cov;
    }

    arma::mat *get_marginal_cov_chol_ptr()
    {
        return &marginal_cov_chol;
    }
};

struct mvn_regression_model_ : public model_base,
                               public regression_base
{
    arma::vec sample_param;

    arma::mat data_cov_inv;

    arma::vec *mean_prior = nullptr;

    arma::mat cov_prior;
    arma::mat cov_prior_inv;
    arma::mat *cov_prior_chol = nullptr;

    arma::vec mean_posterior;
    arma::mat cov_posterior;
    arma::mat cov_posterior_chol;

    arma::mat *constraints = nullptr;

    arma::uvec reorder_idx;
    arma::uvec free_idx;
    arma::uvec fix_idx;

    arma::uword n_free;

    arma::vec fix_coefs;

    arma::mat F_raw;
    arma::mat F_reorder;
    arma::mat F_fixed;
    arma::mat F_free;

    arma::vec z;

    arma::mat D;
    arma::vec d;

    arma::mat pred;

    mvn_regression_model_(
        arma::vec *mean_prior_,
        arma::mat *cov_prior_chol_,
        arma::mat *constraints_)
        : mean_prior(mean_prior_),
          cov_prior_chol(cov_prior_chol_),
          constraints(constraints_)
    {
        coefficient.set_size(constraints->n_rows, constraints->n_cols);

        construct_cov(cov_prior, *cov_prior_chol);
        cov_prior_inv = arma::inv_sympd(cov_prior);

        reorder_idx = reorder_row2col(
            constraints->n_rows,
            constraints->n_cols);

        free_idx = arma::find_nonfinite(*constraints);
        fix_idx = arma::find_finite(*constraints);

        n_free = free_idx.n_elem;

        fix_coefs = constraints->elem(fix_idx);

        sample_param.set_size(n_free);

        D.set_size(n_free, n_free);
        d.set_size(n_free);
    }

    void calc_posterior_parameters() override
    {

        stabalized_inv(data_cov,
                       data_cov_inv,
                       (*data_cov_chol) * data_cov_chol->t());

        D.zeros();
        d.zeros();

        arma::mat identity_mat =
            arma::eye(coefficient.n_rows,
                      coefficient.n_rows);
        for (arma::uword t = 0; t < n_time; t++)
        {
            F_raw = arma::kron(identity_mat,
                               predictor->col(t).t());

            F_reorder = F_raw.cols(reorder_idx);

            F_fixed = F_reorder.cols(fix_idx);
            F_free = F_reorder.cols(free_idx);

            z = outcome->col(t) - data_mean->col(t) - F_fixed * fix_coefs;

            D += F_free.t() * data_cov_inv * F_free;

            d += F_free.t() * data_cov_inv * z;
        }
        D += cov_prior_inv;
        d += cov_prior_inv * (*mean_prior);

        cov_posterior = arma::inv_sympd(D);

        cov_posterior_chol = arma::chol(cov_posterior, "lower");

        mean_posterior = cov_posterior * d;
    }

    void make_predictions()
    {
        pred = *data_mean + coefficient * (*predictor);
    };

    void sample_prior() override
    {
        mat_rnorm(sample_param,
                  *mean_prior,
                  *cov_prior_chol);

        coefficient = vec2mat(sample_param,
                              *constraints);
    }

    void sample_posterior() override
    {
        mat_rnorm(sample_param,
                  mean_posterior,
                  cov_posterior_chol);

        coefficient = vec2mat(sample_param,
                              *constraints);
    }

    double log_marginal_likelihood() override
    {
        return 0.0;
    }

    // Setters
    void set_likelihood_pars(arma::mat *data_mean_, arma::mat *cov_chol_)
    {
        data_mean = data_mean_;
        data_cov_chol = cov_chol_;

        stabalized_inv(data_cov,
                       data_cov_inv,
                       (*data_cov_chol) * data_cov_chol->t());
    };

    // Getters
    arma::mat *get_prediction_ptr()
    {
        return &pred;
    }

    arma::mat get_prediction()
    {
        return pred;
    }
};

#endif