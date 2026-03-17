#ifndef DERIVED_STRUCTS_H
#define DERIVED_STRUCTS_H

// #include <RcppArmadillo.h>
// #include "linear_algebra.h"
#include "base_structs.h"
// #include "imc_gp_struct.h"
// #include "hsgp_struct.h"
// #include <rcpptimer.h>
// #include "timer.h"

// struct mn_iw_model_ : public model_base
// {
//     std::unique_ptr<iw_model_> iw;
//     std::unique_ptr<mn_regression_model> mn;

//     mn_iw_model_(std::unique_ptr<mn_regression_model> mn_model_,
//                  std::unique_ptr<iw_model_> iw_model_)
//         : iw(std::move(iw_model_)),
//           mn(std::move(mn_model_))
//     {
//         iw->data_mean = mn->get_marginal_mean_ptr();
//         iw->data_cov_chol = mn->get_marginal_cov_chol_ptr();
//         mn->set_row_cov(iw->get_cov_chol_ptr());
//     }

//     void calc_posterior_parameters() override
//     {
//         mn->calc_posterior_parameters();
//         // mn->calc_marginal_parameters();
//         iw->calc_posterior_parameters();
//     }

//     void sample_prior() override
//     {
//         iw->sample_prior();
//         mn->sample_prior();
//     }

//     void sample_posterior() override
//     {
//         iw->sample_posterior();
//         mn->sample_posterior();
//     }

//     double log_marginal_likelihood() override
//     {
//         marginal_log_likelihood = iw->log_marginal_likelihood();
//         return marginal_log_likelihood;
//     }

//     // Setters
//     void set_outcome(const arma::mat *outcome_)
//     {
//         model_base::set_outcome(outcome_);
//         iw->set_outcome(outcome);
//         mn->set_outcome(outcome);
//     }

//     void set_likelihood_pars(arma::mat *data_mean_, arma::mat *cov_chol_)
//     {
//         mn->set_likelihood_pars(data_mean_, cov_chol_);
//         iw->set_likelihood_pars(mn->get_marginal_mean_ptr(),
//                                 mn->get_marginal_cov_chol_ptr());
//     };

//     // Getters
//     arma::mat get_cov() const
//     {
//         return iw->get_cov();
//     }

//     arma::mat get_param() const
//     {
//         return mn->get_coefficient();
//     }
// };

struct mn_iw_model_ : public model_base
{
    std::unique_ptr<iw_model_conjugate> iw;
    std::unique_ptr<mn_regression_model> mn;

    mn_iw_model_(std::unique_ptr<mn_regression_model> mn_model_,
                 std::unique_ptr<iw_model_conjugate> iw_model_)
        : iw(std::move(iw_model_)),
          mn(std::move(mn_model_))
    {
        // iw->data_mean = mn->get_marginal_mean_ptr();
        // iw->data_cov_chol = mn->get_marginal_cov_chol_ptr();
        iw->set_mn_prior_pointers(
            mn->coefficient_prior,
            mn->col_cov_prior_chol,
            &mn->col_cov_prior_inv);
        iw->set_mn_posterior_pointers(
            &mn->coefficient_posterior,
            &mn->col_cov_posterior_chol,
            &mn->col_cov_posterior_inv);

        mn->set_row_cov(iw->get_cov_chol_ptr());
    }

    void calc_posterior_parameters() override
    {
        mn->calc_posterior_parameters();
        // mn->calc_marginal_parameters();
        iw->calc_posterior_parameters();
    }

    void sample_prior() override
    {
        iw->sample_prior();
        mn->sample_prior();
    }

    void sample_posterior() override
    {
        iw->sample_posterior();
        mn->sample_posterior();
    }

    double log_marginal_likelihood() override
    {
        marginal_log_likelihood = iw->log_marginal_likelihood();
        return marginal_log_likelihood;
    }

    // Setters
    void set_outcome(const arma::mat *outcome_)
    {
        model_base::set_outcome(outcome_);
        iw->set_outcome(outcome);
        mn->set_outcome(outcome);
    }

    void set_likelihood_pars(arma::mat *data_mean_, arma::mat *cov_chol_)
    {
        mn->set_likelihood_pars(data_mean_, cov_chol_);
        iw->set_likelihood_pars(data_mean_, cov_chol_);
    };

    // Getters
    arma::mat get_cov() const
    {
        return iw->get_cov();
    }

    arma::mat get_param() const
    {
        return mn->get_coefficient();
    }
};

struct mvn_iw_model_ : public model_base
{

    std::unique_ptr<mvn_regression_model_> mvn;
    std::unique_ptr<iw_model_> iw;

    arma::mat data_cov_chol;

    mvn_iw_model_(std::unique_ptr<mvn_regression_model_> mvn_model_,
                  std::unique_ptr<iw_model_> iw_model_)
        : mvn(std::move(mvn_model_)), iw(std::move(iw_model_))
    {
        mvn->data_cov_chol = iw->get_cov_chol_ptr();
        iw->data_mean = mvn->get_prediction_ptr();
        // iw->sample_prior();
    }

    void calc_posterior_parameters() override
    {
        mvn->set_likelihood_pars(mvn->data_mean, iw->get_cov_chol_ptr());
        mvn->calc_posterior_parameters();
    }

    void sample_prior() override
    {
        iw->sample_prior();
        mvn->sample_prior();
    }

    void sample_posterior() override
    {
        mvn->sample_posterior();
        mvn->make_predictions();
        iw->calc_posterior_parameters();
        iw->sample_posterior();
    }

    double log_marginal_likelihood() override
    {
        return 0.0;
    }

    void set_outcome(const arma::mat *outcome_)
    {
        model_base::set_outcome(outcome_);
        data_cov_chol = identity(n_time);
        iw->set_outcome(outcome);
        mvn->set_outcome(outcome);
    }

    // Setters
    void set_likelihood_pars(arma::mat *data_mean_)
    {
        mvn->set_likelihood_pars(data_mean_, iw->get_cov_chol_ptr());
        iw->set_likelihood_pars(mvn->get_prediction_ptr(), &data_cov_chol);
    };

    // Getters
    arma::mat get_cov() const
    {
        return iw->get_cov();
    }

    arma::mat get_param() const
    {
        return mvn->get_coefficient();
    }
};

struct model_wrapper_base
{
    arma::mat *param_ptr = nullptr;

    const arma::mat *predictor = nullptr;
    const arma::mat *covariate = nullptr;

    arma::uword d_pred = 0;
    arma::uword d_covariate = 0;

    arma::mat combined_data;

    const arma::mat *predictor_prior_cov_chol = nullptr;
    const arma::mat *covar_prior_cov_chol = nullptr;

    arma::mat joined_prior_cov_chol;

    model_wrapper_base(
        const arma::mat *predictor_,
        const arma::mat *covariate_,
        const arma::mat *predictor_prior_cov_chol_,
        const arma::mat *covar_prior_cov_chol_)
        : predictor(predictor_),
          covariate(covariate_),
          predictor_prior_cov_chol(predictor_prior_cov_chol_),
          covar_prior_cov_chol(covar_prior_cov_chol_)
    {
        d_pred = predictor->n_rows;
        d_covariate = covariate ? covariate->n_rows : 0;

        combine_data();
    }

    void combine_data()
    {
        if (covariate)
            combined_data = arma::join_cols(*predictor, *covariate);
        else
            combined_data = *predictor;
    }

    virtual void combine_priors() = 0;

    void set_param_ptr(arma::mat *param_ptr_)
    {
        param_ptr = param_ptr_;
    }

    arma::mat *get_data_ptr()
    {
        return &combined_data;
    }

    arma::mat *get_prior_cov_chol_ptr()
    {
        return &joined_prior_cov_chol;
    }

    arma::mat get_pred_param() const
    {
        return param_ptr->cols(0, d_pred - 1);
    }

    arma::mat get_covar_param() const
    {
        if (d_covariate == 0)
            return arma::mat(param_ptr->n_rows, 0);

        return param_ptr->cols(d_pred, d_pred + d_covariate - 1);
    }
};

struct mvn_covar_wrapper : public model_wrapper_base
{
    const arma::vec *predictor_prior_mean = nullptr;
    const arma::vec *covar_prior_mean = nullptr;

    arma::vec joined_prior_mean;

    const arma::mat *predictor_constraints = nullptr;
    const arma::mat *covar_constraints = nullptr;

    arma::mat joined_constraints;

    mvn_covar_wrapper(
        const arma::mat *predictor_,
        const arma::mat *covariate_,
        const arma::mat *predictor_constraints_,
        const arma::mat *covar_constraints_,
        const arma::vec *predictor_prior_mean_,
        const arma::vec *covar_prior_mean_,
        const arma::mat *predictor_prior_cov_chol_,
        const arma::mat *covar_prior_cov_chol_)
        : model_wrapper_base(
              predictor_,
              covariate_,
              predictor_prior_cov_chol_,
              covar_prior_cov_chol_),
          predictor_constraints(predictor_constraints_),
          covar_constraints(covar_constraints_),
          predictor_prior_mean(predictor_prior_mean_),
          covar_prior_mean(covar_prior_mean_)
    {
        combine_priors();
    };

    void combine_priors() override
    {

        if (d_covariate == 0)
        {
            joined_prior_mean = *predictor_prior_mean;
            joined_prior_cov_chol = *predictor_prior_cov_chol;
            joined_constraints = *predictor_constraints;
        }
        else
        {
            joined_prior_mean = arma::join_cols(*predictor_prior_mean,
                                                *covar_prior_mean);
            joined_prior_cov_chol = diag_join(*predictor_prior_cov_chol,
                                              *covar_prior_cov_chol);
            joined_constraints = arma::join_rows(*predictor_constraints,
                                                 *covar_constraints);
        }
    }

    arma::vec *get_mean_ptr()
    {
        return &joined_prior_mean;
    }

    arma::mat *get_constraints_ptr()
    {
        return &joined_constraints;
    }
};

struct mn_covar_wrapper : public model_wrapper_base
{
    const arma::mat *predictor_prior_mean = nullptr;
    const arma::mat *covar_prior_mean = nullptr;

    arma::mat joined_prior_mean;

    mn_covar_wrapper(
        const arma::mat *predictor_,
        const arma::mat *covariate_,
        const arma::mat *predictor_prior_mean_,
        const arma::mat *covar_prior_mean_,
        const arma::mat *predictor_prior_cov_chol_,
        const arma::mat *covar_prior_cov_chol_)
        : model_wrapper_base(
              predictor_,
              covariate_,
              predictor_prior_cov_chol_,
              covar_prior_cov_chol_),
          predictor_prior_mean(predictor_prior_mean_),
          covar_prior_mean(covar_prior_mean_)
    {
        combine_priors();
    }

    void combine_priors() override
    {
        if (d_covariate == 0)
        {
            joined_prior_mean = *predictor_prior_mean;
            joined_prior_cov_chol = *predictor_prior_cov_chol;
        }
        else
        {
            joined_prior_mean = arma::join_rows(*predictor_prior_mean,
                                                *covar_prior_mean);
            joined_prior_cov_chol = diag_join(*predictor_prior_cov_chol,
                                              *covar_prior_cov_chol);
        };
    }

    arma::mat *get_mean_ptr()
    {
        return &joined_prior_mean;
    }
};

#endif