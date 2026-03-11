#ifndef DERIVED_STRUCTS_H
#define DERIVED_STRUCTS_H

#include <RcppArmadillo.h>
#include "linear_algebra.h"
#include "base_structs.h"
#include "imc_gp_class.h"
#include "hsgp_class.h"
#include <rcpptimer.h>
#include "timer.h"

struct mn_iw_model_ : public model_base
{
    std::unique_ptr<iw_model_> iw;
    std::unique_ptr<mn_conjugate_base> mn;

    mn_iw_model_(std::unique_ptr<mn_conjugate_base> mn_model_,
                 std::unique_ptr<iw_model_> iw_model_)
        : iw(std::move(iw_model_)),
          mn(std::move(mn_model_)) {};

    void calc_posterior_parameters() override
    {
        mn->calc_posterior_parameters();
        // iw.set_data_mean(mn.get_marginal_mean());
        // iw.set_data_col_cov(mn.get_marginal_col_cov());
        iw->calc_posterior_parameters();
    };

    void sample_prior() override
    {
        iw->sample_prior();
        mn->set_row_cov(iw->get_cov());
        mn->sample_prior();
    };

    void sample_posterior() override
    {
        iw->sample_posterior();
        mn->set_row_cov(iw->get_cov());
        mn->sample_posterior();
    };

    double log_marginal_likelihood() override
    {
        marginal_log_likelihood = iw->log_marginal_likelihood();
        return marginal_log_likelihood;
    };

    // Setters
    void set_data(const Dataset &data,
                  const arma::mat &data_mean_,
                  const arma::mat &data_cov_)
    {
        // Doing anything or am I just copying matrices for fun?
        // model_base::set_daeta(data, data_mean_, data_cov_);
        timer.tic("mh.set_data.mnmn");
        mn->set_data(data, data_mean_, data_cov_);
        timer.toc("mh.set_data.mnmn");
        timer.tic("mh.set_data.iw");
        iw->set_data(data, mn->get_marginal_mean(),
                     mn->get_marginal_cov());
        timer.toc("mh.set_data.iw");
    };

    // Getters
    arma::mat get_cov()
    {
        return iw->get_cov();
    };

    arma::mat get_param()
    {
        return mn->get_param();
    };
};

struct mn_mn_model_ : public mn_conjugate_base
{
    std::unique_ptr<mn_conjugate_base> mn1;
    std::unique_ptr<mn_conjugate_base> mn2;

    mn_mn_model_(std::unique_ptr<mn_conjugate_base> mn1_model_,
                 std::unique_ptr<mn_conjugate_base> mn2_model_)
        : mn1(std::move(mn1_model_)),
          mn2(std::move(mn2_model_)) {};

    void calc_posterior_parameters() override
    {
        mn1->calc_posterior_parameters();
        mn2->calc_posterior_parameters();
    };

    void sample_prior() override
    {
        mn1->sample_prior();
        mn2->sample_prior();
    };

    void sample_posterior() override
    {
        mn2->sample_posterior();
        mn1->sample_posterior();
    };

    double log_marginal_likelihood() override
    {
        marginal_log_likelihood = mn2->log_marginal_likelihood();
        return marginal_log_likelihood;
    };

    // Setters
    void set_data(const Dataset &data,
                  const arma::mat &data_mean_,
                  const arma::mat &data_cov_)
    {
        timer.tic("mh.set_data.mn1Test");
        mn1->set_data(data, data_mean_, data_cov_);
        timer.toc("mh.set_data.mn1Test");
        timer.tic("mh.set_data.mn2");
        mn2->set_data(data, mn1->get_marginal_mean(),
                      mn1->get_marginal_cov());
        timer.toc("mh.set_data.mn2");
    };

    void set_row_cov(const arma::mat &row_cov_) override
    {
        mn1->set_row_cov(row_cov_);
        mn2->set_row_cov(row_cov_);
    };

    // Getters
    arma::mat get_param1()
    {
        return mn1->get_param();
    };

    arma::mat get_param2()
    {
        return mn2->get_param();
    };

    arma::mat get_marginal_mean() override
    {
        return mn2->get_marginal_mean();
    }

    arma::mat get_marginal_cov() override
    {
        return mn2->get_marginal_cov();
    }
};

struct mvn_iw_model_ : public model_base
{
    std::unique_ptr<iw_model_> iw;
    std::unique_ptr<mvn_regression_model_> mvn;

    Dataset data;

    mvn_iw_model_(std::unique_ptr<mvn_regression_model_> mvn_model_,
                  std::unique_ptr<iw_model_> iw_model_)
        : iw(std::move(iw_model_)),
          mvn(std::move(mvn_model_))
    {
        iw->sample_prior();
    };

    void calc_posterior_parameters() override
    {
        mvn->set_data(data, data_mean, iw->get_cov());
        mvn->calc_posterior_parameters();
    };

    void sample_prior() override
    {
        iw->sample_prior();
        mvn->sample_prior();
    };

    void sample_posterior() override
    {
        mvn->sample_posterior();
        iw->set_data(data, mvn->get_param() * *mvn->predictor,
                     identity(n_time));
        iw->calc_posterior_parameters();
        iw->sample_posterior();
    };

    double log_marginal_likelihood() override
    {
        return 0;
    };

    // Setters
    void set_data(const Dataset &data,
                  const arma::mat &data_mean_)
    {
        this->data = data;
        outcome = data.outcome;
        data_mean = data_mean_;

        n_time = outcome->n_cols;
        d_outcome = outcome->n_rows;

        mvn->set_data(data, data_mean_, iw->get_cov());
        iw->set_data(data,
                     mvn->get_param() * *mvn->predictor, identity(n_time));
    };

    // Getters
    arma::mat get_cov()
    {
        return iw->get_cov();
    };

    arma::mat get_param()
    {
        return mvn->get_param();
    };
};

struct mn_gp_mean_model_ : public mn_mean_model_, public gp_base, public regression_base
{
    std::unique_ptr<imc_gp> gp;

    mn_gp_mean_model_(std::unique_ptr<imc_gp> gp_,
                      const arma::mat &row_cov_,
                      size_t predictor_idx_)
        : mn_mean_model_(arma::mat(), row_cov_, arma::mat()),
          regression_base(predictor_idx_), gp(std::move(gp_))
    {
        set_row_cov(row_cov_);
    };

    void calc_posterior_parameters() override
    {
        gp->update_test_data(*predictor);
        gp->compute_predictive(false);
        param_posterior = gp->pred_mean;
        col_cov_posterior_chol = gp->pred_col_cov_chol;
        col_cov_posterior = col_cov_posterior_chol * col_cov_posterior_chol.t();
    };

    void sample_prior() override
    {
        set_param_prior();
        set_col_cov_prior();
        rmatnorm(param, param_prior,
                 row_cov_chol, col_cov_prior_chol);
    };

    // Setters
    void set_hyperparameters(double alpha, double rho) override
    {
        gp->update_hyperparameters(alpha, rho);
        set_col_cov_prior();
    };

    void set_row_cov(const arma::mat &row_cov_) override
    {
        row_cov = row_cov_;
        row_cov_chol = chol(row_cov, "lower");
        gp->update_sigma(row_cov);
    };

    void set_param_prior()
    {
        param_prior = gp->train_mu;
    };

    void set_col_cov_prior()
    {
        col_cov_prior_chol = gp->train_k_chol;
        col_cov_prior = gp->train_k_chol * gp->train_k_chol.t();
    };

    arma::mat get_gp_predictions() override
    {
        return get_param();
    };

    void set_data(const Dataset &data,
                  const arma::mat &data_mean_,
                  const arma::mat &data_cov_)
    {
        mn_mean_model_::set_data(data, data_mean_, data_cov_);
        predictor = data.predictors[predictor_idx];
        gp->update_train_data(*predictor, *outcome - data_mean);
        gp->set_train_y_cov(data_cov_);
        set_param_prior();
        set_col_cov_prior();
    };
};

struct mn_hsgp_regression_model_ : public mn_regression_model_, public gp_base
{
    std::unique_ptr<hsgp_approx> hsgp;

    mn_hsgp_regression_model_(
        std::unique_ptr<hsgp_approx> hsgp_,
        const arma::mat &row_cov_,
        size_t predictor_idx_)
        : mn_regression_model_(arma::mat(),
                               arma::mat(),
                               row_cov_,
                               predictor_idx_),
          hsgp(std::move(hsgp_))
    {

        arma::uword d_cols = hsgp->indices.n_rows;
        arma::uword d_rows = row_cov_.n_rows;

        param_prior.zeros(d_rows, d_cols);
        param.set_size(d_rows, d_cols);
        param_posterior.set_size(d_rows, d_cols);

        set_col_cov_prior();
        col_cov_posterior.set_size(d_cols, d_cols);
        col_cov_posterior_chol.set_size(d_cols, d_cols);
    };

    void set_hyperparameters(double alpha, double rho) override
    {
        hsgp->update_hyperparameters(alpha, rho);
        set_col_cov_prior();
    };

    void set_col_cov_prior()
    {
        mn_regression_model_::set_col_cov_prior(hsgp->scale());
    };

    arma::mat get_gp_predictions() override
    {
        return get_param() * *predictor;
    };

    void set_data(const Dataset &data,
                  const arma::mat &data_mean_,
                  const arma::mat &data_cov_)
    {
        hsgp->phi_transform(*data.predictors[predictor_idx]);
        predictor = &hsgp->phi;
        d_predictor = predictor->n_rows;

        // Data cov is currently fixed to be I
        mn_conjugate_base::set_data(data, data_mean_, data_cov_);
    };
};

struct model_wrapper_base
{
    arma::mat *param_ptr = nullptr;

    arma::mat combined_data;
    const arma::mat *predictor = nullptr;
    const arma::mat *covariate = nullptr;

    arma::uword d_pred;
    arma::uword d_covariate;

    arma::mat joined_prior_cov;
    arma::mat predictor_prior_cov;
    arma::mat covar_prior_cov;

    model_wrapper_base(
        const arma::mat *predictor_,
        const arma::mat *covariate_,
        const arma::mat &predictor_prior_cov_,
        const arma::mat &covar_prior_cov_)
        : predictor(predictor_),
          covariate(covariate_),
          predictor_prior_cov(predictor_prior_cov_),
          covar_prior_cov(covar_prior_cov_)
    {
        d_pred = predictor_->n_rows;
        d_covariate = covariate_ ? covariate_->n_rows : 0;

        if (d_covariate == 0)
            joined_prior_cov = predictor_prior_cov;
        else
            joined_prior_cov =
                diag_join(predictor_prior_cov, covar_prior_cov);

        set_data();
    }

    void set_data()
    {
        combined_data = arma::join_cols(*predictor, *covariate);
    }

    void set_param_ptr(arma::mat *param_ptr_)
    {
        param_ptr = param_ptr_;
    }

    const arma::mat &get_cov() const
    {
        return joined_prior_cov;
    }

    arma::mat get_pred_param()
    {
        return param_ptr->cols(0, d_pred - 1);
    }

    arma::mat get_covar_param()
    {
        if (d_covariate == 0)
            return arma::mat(param_ptr->n_rows, 0);

        return param_ptr->cols(d_pred, d_pred + d_covariate - 1);
    }
};

struct mvn_covar_wrapper : public model_wrapper_base
{
    arma::mat joined_mat_const;
    arma::mat predictor_mat_const;
    arma::mat covar_mat_const;

    arma::vec joined_prior_mean;
    arma::vec predictor_prior_mean;
    arma::vec covar_prior_mean;

    mvn_covar_wrapper(
        const arma::mat *predictor_,
        const arma::mat *covariate_,
        const arma::mat &predictor_mat_const_,
        const arma::mat &covar_mat_const_,
        const arma::vec &predictor_prior_mean_,
        const arma::vec &covar_prior_mean_,
        const arma::mat &predictor_prior_cov_,
        const arma::mat &covar_prior_cov_)
        : model_wrapper_base(
              predictor_,
              covariate_,
              predictor_prior_cov_,
              covar_prior_cov_),
          predictor_mat_const(predictor_mat_const_),
          covar_mat_const(covar_mat_const_),
          predictor_prior_mean(predictor_prior_mean_),
          covar_prior_mean(covar_prior_mean_)
    {
        joined_mat_const =
            arma::join_rows(predictor_mat_const, covar_mat_const);

        joined_prior_mean =
            arma::join_cols(predictor_prior_mean, covar_prior_mean);
    }

    const arma::mat &get_const() const
    {
        return joined_mat_const;
    }

    const arma::vec &get_mean() const
    {
        return joined_prior_mean;
    }
};

struct mn_covar_wrapper : public model_wrapper_base
{
    arma::mat joined_mat_const;
    arma::mat predictor_mat_const;
    arma::mat covar_mat_const;

    arma::mat joined_prior_mean;
    arma::mat predictor_prior_mean;
    arma::mat covar_prior_mean;

    mn_covar_wrapper(
        const arma::mat *predictor_,
        const arma::mat *covariate_,
        const arma::mat &predictor_prior_mean_,
        const arma::mat &covar_prior_mean_,
        const arma::mat &predictor_prior_cov_,
        const arma::mat &covar_prior_cov_)
        : model_wrapper_base(
              predictor_,
              covariate_,
              predictor_prior_cov_,
              covar_prior_cov_),
          predictor_prior_mean(predictor_prior_mean_),
          covar_prior_mean(covar_prior_mean_)
    {
        joined_prior_mean =
            arma::join_rows(predictor_prior_mean, covar_prior_mean);
    }

    const arma::mat &get_mean() const
    {
        return joined_prior_mean;
    }
};

#endif