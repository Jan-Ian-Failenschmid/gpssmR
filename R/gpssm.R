### Define classes, generics and methods ---------------------------------------
# Design matrix ----------------------------------------------------------------
gpssm_design_mat <- R6::R6Class("gpssm_design_mat",
  private = list(
    dim = NULL,
    par_vec = NULL,
    par_names = NULL,
    alt = FALSE
  ),
  public = list(
    name = NULL,
    value = NULL,
    constraint = NULL,

    ## original formulation
    prior_mean = NULL,
    prior_col_cov = NULL,

    ## alternative formulation
    prior_mean_alt = NULL,
    prior_cov_alt = NULL,
    initialize = function(name,
                          constraint,
                          prior_mean = NULL,
                          prior_col_cov = NULL,
                          prior_mean_alt = NULL,
                          prior_cov_alt = NULL) {
      stopifnot(is.character(name), length(name) == 1, nchar(name) > 0)
      stopifnot(is.matrix(constraint))

      ## exactly one formulation must be provided
      use_std <- !is.null(prior_mean) || !is.null(prior_col_cov)
      use_alt <- !is.null(prior_mean_alt) || !is.null(prior_cov_alt)
      stopifnot(xor(use_std, use_alt))

      if (use_std) {
        stopifnot(
          is.matrix(prior_mean),
          is.matrix(prior_col_cov)
        )
        private$alt <- FALSE
        self$prior_mean <- prior_mean
        self$prior_col_cov <- prior_col_cov
        self$prior_mean_alt <- numeric(0)
        self$prior_cov_alt <- matrix(numeric(0), 0, 0)
      } else {
        stopifnot(
          is.vector(prior_mean_alt),
          is.matrix(prior_cov_alt)
        )

        n_free <- sum(is.na(constraint))

        stopifnot(
          length(prior_mean_alt) == n_free,
          nrow(prior_cov_alt) == n_free,
          ncol(prior_cov_alt) == n_free
        )

        private$alt <- TRUE
        self$prior_mean_alt <- prior_mean_alt
        self$prior_cov_alt <- prior_cov_alt

        self$prior_mean <- matrix(numeric(0), 0, 0)
        self$prior_col_cov <- matrix(numeric(0), 0, 0)
      }

      self$name <- name
      self$constraint <- constraint

      private$dim <- dim(constraint)
      self$value <- constraint
      private$par_vec <- as.vector(self$value)

      idx <- expand.grid(
        row = seq_len(private$dim[1]),
        col = seq_len(private$dim[2])
      )

      private$par_names <- apply(
        idx, 1,
        function(x) paste0(name, "[", x[1], ",", x[2], "]")
      )
    },
    get_dim = function() {
      private$dim
    },
    get_par_vec = function() {
      private$par_vec
    },
    get_par_names = function() {
      private$par_names
    },
    uses_alt_prior = function() {
      private$alt
    }
  )
)

gpssm_mn_prior <- R6::R6Class("gpssm_mn_prior",
  private = list(
    dim = NULL,
    par_vec = NULL,
    par_names = NULL
  ),
  public = list(
    name = NULL,
    value = NULL,
    prior_mean = NULL,
    prior_col_cov = NULL,
    initialize = function(name, prior_mean = NULL, prior_col_cov = NULL) {
      stopifnot(is.character(name), length(name) == 1, nchar(name) > 0)
      stopifnot(is.matrix(prior_mean), is.matrix(prior_col_cov))

      self$prior_mean <- prior_mean
      self$prior_col_cov <- prior_col_cov
      self$name <- name

      private$dim <- dim(prior_mean)
      self$value <- prior_mean
      private$par_vec <- as.vector(self$value)

      idx <- expand.grid(
        row = seq_len(private$dim[1]),
        col = seq_len(private$dim[2])
      )

      private$par_names <- apply(
        idx, 1,
        function(x) paste0(name, "[", x[1], ",", x[2], "]")
      )
    },
    get_dim = function() {
      private$dim
    },
    get_par_vec = function() {
      private$par_vec
    },
    get_par_names = function() {
      private$par_names
    }
  )
)

gpssm_mvn_prior <- R6::R6Class("gpssm_mvn_prior",
  private = list(
    dim = NULL,
    par_vec = NULL,
    par_names = NULL,
    alt = FALSE
  ),
  public = list(
    name = NULL,
    value = NULL,
    constraint = NULL,

    ## alternative formulation
    prior_mean = NULL,
    prior_cov = NULL,
    initialize = function(name,
                          constraint,
                          prior_mean = NULL,
                          prior_cov = NULL) {
      stopifnot(is.character(name), length(name) == 1, nchar(name) > 0)
      stopifnot(is.matrix(constraint))
      stopifnot(is.vector(prior_mean), is.matrix(prior_cov))

      n_free <- sum(is.na(constraint))

      stopifnot(
        length(prior_mean) == n_free,
        nrow(prior_cov) == n_free,
        ncol(prior_cov) == n_free
      )

      self$prior_mean <- prior_mean
      self$prior_cov <- prior_cov

      self$name <- name
      self$constraint <- constraint

      private$dim <- dim(constraint)
      self$value <- vec2mat(prior_mean, constraint)
      private$par_vec <- as.vector(self$value)

      idx <- expand.grid(
        row = seq_len(private$dim[1]),
        col = seq_len(private$dim[2])
      )

      private$par_names <- apply(
        idx, 1,
        function(x) paste0(name, "[", x[1], ",", x[2], "]")
      )
    },
    get_dim = function() {
      private$dim
    },
    get_par_vec = function() {
      private$par_vec
    },
    get_par_names = function() {
      private$par_names
    }
  )
)

# gpssm_design_mat$new(
#   name = "B",
#   constraint = matrix(NA, 1, M_basis),
#   prior_mean = matrix(0, 1, M_basis),
#   prior_col_cov = diag(1, M_basis) # Gets overwritten in c++
# )

# Covariance matrix ------------------------------------------------------------
gpssm_cov_mat <- R6::R6Class("gpssm_cov_mat",
  private = list(
    dim = NULL,
    par_vec = NULL,
    par_names = NULL
  ),
  public = list(
    name = NULL,
    value = NULL,
    prior_df = NULL,
    prior_scale = NULL,
    initialize = function(name, prior_df, prior_scale) {
      stopifnot(is.character(name), length(name) == 1, nchar(name) > 0)
      stopifnot(is.numeric(prior_df), length(prior_df) == 1)
      stopifnot(is.matrix(prior_scale), nrow(prior_scale) == ncol(prior_scale))
      stopifnot(is_psd_chol(prior_scale))

      private$dim <- nrow(prior_scale)
      stopifnot(prior_df >= private$dim)

      self$name <- name
      self$prior_df <- prior_df
      self$prior_scale <- prior_scale

      self$value <- diag(1, private$dim)
      private$par_vec <- as.vector(self$value)
      idx <- expand.grid(i = 1:private$dim, j = 1:private$dim)
      private$par_names <- apply(
        idx, 1,
        function(x) paste0(name, "[", x[1], ",", x[2], "]")
      )
    },
    validate = function() {
      if (!is_psd_chol(self$value)) {
        stop("'value' must be positive semi-definite.")
      }
    },
    get_dim = function() {
      private$dim
    },
    get_par_vec = function() {
      private$par_vec
    },
    get_par_names = function() {
      private$par_names
    }
  )
)

# gpssm_cov_mat$new(
#     name = "Q",
#     prior_df = 2,
#     prior_scale = diag(2, 1)
# )

# Data class -------------------------------------------------------------------
gpssm_data <- R6::R6Class("gpssm_data",
  private = list(
    time = NULL,
    d_lat = NULL,
    d_obs = NULL,
    d_covariate_meas = NULL,
    d_covariate_dyn = NULL,
    data_name_vec = NULL,
    latent_name_vec = NULL,
    covariate_meas_name_vec = NULL,
    covariate_dyn_name_vec = NULL
  ),
  public = list(
    data = NULL,
    latent_start = NULL,
    covariate_meas = NULL,
    covariate_dyn = NULL,
    t0_mean = NULL,
    t0_cov = NULL,
    data_name = NULL,
    latent_name = NULL,
    covariate_meas_name = NULL,
    covariate_dyn_name = NULL,
    initialize = function(data,
                          latent_start,
                          covariate_meas = NULL,
                          covariate_dyn = NULL,
                          t0_mean,
                          t0_cov,
                          data_name,
                          latent_name,
                          covariate_meas_name = NULL,
                          covariate_dyn_name = NULL) {
      stopifnot(
        is.matrix(data),
        is.matrix(latent_start) || is.list(latent_start)
      )

      if (is.null(covariate_meas_name)) {
        covariate_meas_name <- "covar_meas"
      }
      if (is.null(covariate_dyn_name)) {
        covariate_dyn_name <- "covar_dyn"
      }

      ## if only one covariate is provided, copy it to the other
      if (is.null(covariate_meas)) {
        covariate_meas <- matrix(numeric(0), 0, ncol(data))
      }
      if (is.null(covariate_dyn)) {
        covariate_dyn <- matrix(numeric(0), 0, ncol(data))
      }

      stopifnot(
        is.matrix(covariate_meas),
        is.matrix(covariate_dyn)
      )

      stopifnot(
        is.character(data_name),
        is.character(latent_name),
        is.character(covariate_meas_name),
        is.character(covariate_dyn_name)
      )

      self$data <- data
      self$latent_start <- latent_start
      self$covariate_meas <- covariate_meas
      self$covariate_dyn <- covariate_dyn
      self$t0_mean <- t0_mean
      self$t0_cov <- t0_cov
      self$data_name <- data_name
      self$latent_name <- latent_name
      self$covariate_meas_name <- covariate_meas_name
      self$covariate_dyn_name <- covariate_dyn_name

      private$time <- ncol(data)
      if (is.list(latent_start)) {
        private$d_lat <- nrow(latent_start[[1]])
      } else {
        private$d_lat <- nrow(latent_start)
      }

      private$d_obs <- nrow(data)
      private$d_covariate_meas <- nrow(covariate_meas)
      private$d_covariate_dyn <- nrow(covariate_dyn)

      private$data_name_vec <- as.vector(sapply(
        1:private$time,
        function(x) paste0(data_name, "[", 1:private$d_obs, ",", x, "]")
      ))

      private$latent_name_vec <- as.vector(sapply(
        1:private$time,
        function(x) paste0(latent_name, "[", 1:private$d_lat, ",", x, "]")
      ))

      if (private$d_covariate_meas > 0) {
        private$covariate_meas_name_vec <- as.vector(sapply(
          1:private$time,
          function(x) {
            paste0(
              covariate_meas_name,
              "[", 1:private$d_covariate_meas, ",", x, "]"
            )
          }
        ))
      } else {
        private$covariate_meas_name_vec <- character(0)
      }

      if (private$d_covariate_dyn > 0) {
        private$covariate_dyn_name_vec <- as.vector(sapply(
          1:private$time,
          function(x) {
            paste0(
              covariate_dyn_name,
              "[", 1:private$d_covariate_dyn, ",", x, "]"
            )
          }
        ))
      } else {
        private$covariate_dyn_name_vec <- character(0)
      }
    },
    get_time = function() {
      private$time
    },
    get_d_lat = function() {
      private$d_lat
    },
    get_d_obs = function() {
      private$d_obs
    },
    get_d_covariate_meas = function() {
      private$d_covariate_meas
    },
    get_d_covariate_dyn = function() {
      private$d_covariate_dyn
    },
    get_data_name_vec = function() {
      private$data_name_vec
    },
    get_latent_name_vec = function() {
      private$latent_name_vec
    },
    get_covariate_meas_name_vec = function() {
      private$covariate_meas_name_vec
    },
    get_covariate_dyn_name_vec = function() {
      private$covariate_dyn_name_vec
    }
  )
)

# gpssm_data$new(
#   data = y,
#   latent_start = as.matrix(rbind(df$y_obs1)),
#   covariate = matrix(numeric(0), 0, 200),
#   data_name = "Y",
#   latent_name = "X",
#   covariate_name = "Covar"
# )


# Full Model class -------------------------------------------------------------
gpssm <- R6::R6Class("gpssm",
  public = list(
    data = NULL,
    basis_fun_index = NULL,
    boundry_factor = NULL,
    hyperparameter_names = NULL,
    gp_name_vec = NULL,
    dprior = NULL,
    rprior = NULL,
    dyn_design_mat = NULL,
    dyn_covariate_mat = NULL,
    dyn_cov_mat = NULL,
    meas_design_mat = NULL,
    meas_covariate_mat = NULL,
    meas_cov_mat = NULL,
    prior_samples = NULL,
    posterior_samples = NULL,
    initialize = function(data, gp_model, dynamic_model, measurement_model) {
      # Construct GP variable names
      gp_name_vec <- as.vector(sapply(1:(data$get_time() - 1), function(x) {
        paste0(gp_model$gp_name, "[", 1:data$get_d_lat(), ",", x, "]")
      }))

      self$data <- data
      self$basis_fun_index <- gp_model$basis_fun_index
      self$boundry_factor <- gp_model$boundry_factor
      self$hyperparameter_names <- gp_model$hyperparameter_names
      self$gp_name_vec <- gp_name_vec
      self$dprior <- gp_model$dprior
      self$rprior <- gp_model$rprior

      # Dynamic model components
      self$dyn_design_mat <- dynamic_model$design_mat
      self$dyn_covariate_mat <- dynamic_model$covariate_mat
      self$dyn_cov_mat <- dynamic_model$cov_mat

      # Measurement model components
      self$meas_design_mat <- measurement_model$design_mat
      self$meas_covariate_mat <- measurement_model$covariate_mat
      self$meas_cov_mat <- measurement_model$cov_mat
    },
    prior_sample = function(iter, seed = FALSE, pred = FALSE,
                            chains = 1, parallel = "off",
                            exact = FALSE, disp_prog = TRUE,
                            y_test = matrix(numeric(0), 0, 0)) {
      # Choose parallelization plan
      if (parallel == "off") {
        if (seed) {
          set.seed(seed)
        }

        samples <- lapply(1:chains,
          FUN = function(i) {
            chain_sample <- gpssm_prior_sample(
              n_iter = iter,
              n_time = self$data$get_time(),
              d_lat = self$data$get_d_lat(),
              d_obs = self$data$get_d_obs(),
              covariate_dyn = self$data$covariate_dyn,
              covariate_meas = self$data$covariate_meas,
              t0_mean = self$data$t0_mean,
              t0_cov = self$data$t0_cov,
              basis_fun_index = self$basis_fun_index,
              boundry_factor = self$boundry_factor,
              rprior = self$rprior,
              dyn_design_mat_mean = self$dyn_design_mat$prior_mean,
              dyn_covar_mat_mean = self$dyn_covariate_mat$prior_mean,
              dyn_covar_mat_col_cov = self$dyn_covariate_mat$prior_col_cov,
              dyn_cov_df = self$dyn_cov_mat$prior_df,
              dyn_cov_scale = self$dyn_cov_mat$prior_scale,
              meas_design_mat_const = self$meas_design_mat$constraint,
              meas_design_mat_mean_alt = self$meas_design_mat$prior_mean,
              meas_design_mat_cov_alt = self$meas_design_mat$prior_cov,
              meas_covar_mat_const = self$meas_covariate_mat$constraint,
              meas_covar_mat_mean_alt = self$meas_covariate_mat$prior_mean,
              meas_covar_mat_cov_alt = self$meas_covariate_mat$prior_cov,
              meas_cov_df = self$meas_cov_mat$prior_df,
              meas_cov_scale = self$meas_cov_mat$prior_scale,
              y = y_test,
              exact = exact,
              pred = pred,
              disp_prog = disp_prog
            )

            # Assign column names
            colnames(chain_sample) <- c(
              if (pred) self$data$get_data_name_vec(),
              self$data$get_latent_name_vec(),
              self$gp_name_vec,
              self$hyperparameter_names,
              if (!exact) self$dyn_design_mat$get_par_names(),
              self$dyn_covariate_mat$get_par_names(),
              self$dyn_cov_mat$get_par_names(),
              self$meas_design_mat$get_par_names(),
              self$meas_covariate_mat$get_par_names(),
              self$meas_cov_mat$get_par_names(),
              "log_lik"
            )
            chain_sample
          }
        )
      } else if (parallel == "future") {
        if (!require(future.apply)) {
          stop("'future.apply' required for parallel sampling. Please install 'future' or choose the option parallel = 'off'")
        }
        self_local <- self # Local copy os self to be exported to workers
        # Run parallel chains
        samples <- future.apply::future_lapply(1:chains,
          future.seed = seed,
          future.globals = list(
            self_local = self_local,
            gpssm_sample = gpssm_prior_sample
          ),
          FUN = function(i) {
            chain_sample <- gpssm_prior_sample(
              n_iter = iter,
              n_time = self$data$get_time(),
              d_lat = self$data$get_d_lat(),
              d_obs = self$data$get_d_obs(),
              covariate_dyn = self$data$covariate_dyn,
              covariate_meas = self$data$covariate_meas,
              t0_mean = self$data$t0_mean,
              t0_cov = self$data$t0_cov,
              basis_fun_index = self$basis_fun_index,
              boundry_factor = self$boundry_factor,
              rprior = self$rprior,
              dyn_design_mat_mean = self$dyn_design_mat$prior_mean,
              dyn_covar_mat_mean = self$dyn_covariate_mat$prior_mean,
              dyn_covar_mat_col_cov = self$dyn_covariate_mat$prior_col_cov,
              dyn_cov_df = self$dyn_cov_mat$prior_df,
              dyn_cov_scale = self$dyn_cov_mat$prior_scale,
              meas_design_mat_const = self$meas_design_mat$constraint,
              meas_design_mat_mean_alt = self$meas_design_mat$prior_mean,
              meas_design_mat_cov_alt = self$meas_design_mat$prior_cov,
              meas_covar_mat_const = self$meas_covariate_mat$constraint,
              meas_covar_mat_mean_alt = self$meas_covariate_mat$prior_mean,
              meas_covar_mat_cov_alt = self$meas_covariate_mat$prior_cov,
              meas_cov_df = self$meas_cov_mat$prior_df,
              meas_cov_scale = self$meas_cov_mat$prior_scale,
              y = y_test,
              exact = exact,
              pred = pred,
              disp_prog = disp_prog
            )

            # Assign column names
            colnames(chain_sample) <- c(
              if (pred) self$data$get_data_name_vec(),
              self$data$get_latent_name_vec(),
              self$gp_name_vec,
              self$hyperparameter_names,
              if (!exact) self$dyn_design_mat$get_par_names(),
              self$dyn_covariate_mat$get_par_names(),
              self$dyn_cov_mat$get_par_names(),
              self$meas_design_mat$get_par_names(),
              self$meas_covariate_mat$get_par_names(),
              self$meas_cov_mat$get_par_names(),
              "log_lik"
            )
            chain_sample
          }
        )
      }

      self$prior_samples <- posterior::as_draws(samples)

      return(invisible(self))
    },
    sample = function(iter, warm_up, particles, seed = FALSE,
                      thin = 1,
                      chains = 1,
                      mh_adapt_start = 75,
                      mh_rep = 10,
                      pg_rep = 3,
                      parallel = "off",
                      exact = FALSE,
                      post_pred = FALSE,
                      disp_prog = TRUE) {
      # Choose parallelization plan
      if (parallel == "off") {
        if (seed) {
          set.seed(seed)
        }
        samples <- lapply(1:chains,
          FUN = function(i) {
            if (is.list(self$data$latent_start)) {
              latent_start_i <- self$data$latent_start[[i]]
            } else {
              latent_start_i <- self$data$latent_start
            }
            chain_sample <- gpssm_sample(
              n_iter = iter,
              n_warm_up = warm_up,
              n_thin = thin,
              n_particles = particles,
              n_time = self$data$get_time(),
              d_lat = self$data$get_d_lat(),
              d_obs = self$data$get_d_obs(),
              y = self$data$data,
              x = latent_start_i,
              covariate_dyn = self$data$covariate_dyn,
              covariate_meas = self$data$covariate_meas,
              t0_mean = self$data$t0_mean,
              t0_cov = self$data$t0_cov,
              basis_fun_index = self$basis_fun_index,
              boundry_factor = self$boundry_factor,
              dprior = self$dprior,
              rprior = self$rprior,
              dyn_design_mat_mean = self$dyn_design_mat$prior_mean,
              dyn_covar_mat_mean = self$dyn_covariate_mat$prior_mean,
              dyn_covar_mat_col_cov = self$dyn_covariate_mat$prior_col_cov,
              dyn_cov_df = self$dyn_cov_mat$prior_df,
              dyn_cov_scale = self$dyn_cov_mat$prior_scale,
              meas_design_mat_const = self$meas_design_mat$constraint,
              meas_design_mat_mean_alt = self$meas_design_mat$prior_mean,
              meas_design_mat_cov_alt = self$meas_design_mat$prior_cov,
              meas_covar_mat_const = self$meas_covariate_mat$constraint,
              meas_covar_mat_mean_alt = self$meas_covariate_mat$prior_mean,
              meas_covar_mat_cov_alt = self$meas_covariate_mat$prior_cov,
              meas_cov_df = self$meas_cov_mat$prior_df,
              meas_cov_scale = self$meas_cov_mat$prior_scale,
              mh_rep = mh_rep,
              pg_rep = pg_rep,
              mh_adapt_start = mh_adapt_start,
              exact = exact,
              post_pred = post_pred,
              disp_prog = disp_prog
            )


            # Assign column names
            name_parts <- c(
              if (post_pred) self$data$get_data_name_vec(),
              self$data$get_latent_name_vec(),
              self$gp_name_vec,
              self$hyperparameter_names,
              if (!exact) self$dyn_design_mat$get_par_names(),
              self$dyn_covariate_mat$get_par_names(),
              self$dyn_cov_mat$get_par_names(),
              self$meas_design_mat$get_par_names(),
              self$meas_covariate_mat$get_par_names(),
              self$meas_cov_mat$get_par_names(), 
              "log_lik"
            )
            colnames(chain_sample) <- name_parts

            chain_sample
          }
        )
      } else if (parallel == "future") {
        if (!require(future.apply)) {
          stop("'future.apply' required for parallel sampling. Please install 'future' or choose the option parallel = 'off'")
        }
        self_local <- self # Local copy os self to be exported to workers
        # Run parallel chains
        samples <- future.apply::future_lapply(1:chains,
          future.seed = seed,
          future.globals = list(
            self_local = self_local,
            gpssm_sample = gpssm_sample
          ),
          FUN = function(i) {
            if (is.list(self$data$latent_start)) {
              latent_start_i <- self$data$latent_start[[i]]
            } else {
              latent_start_i <- self$data$latent_start
            }
            chain_sample <- gpssm_sample(
              n_iter = iter,
              n_warm_up = warm_up,
              n_thin = thin,
              n_particles = particles,
              n_time = self$data$get_time(),
              d_lat = self$data$get_d_lat(),
              d_obs = self$data$get_d_obs(),
              y = self$data$data,
              x = latent_start_i,
              covariate_dyn = self$data$covariate_dyn,
              covariate_meas = self$data$covariate_meas,
              t0_mean = self$data$t0_mean,
              t0_cov = self$data$t0_cov,
              basis_fun_index = self$basis_fun_index,
              boundry_factor = self$boundry_factor,
              dprior = self$dprior,
              rprior = self$rprior,
              dyn_design_mat_mean = self$dyn_design_mat$prior_mean,
              dyn_covar_mat_mean = self$dyn_covariate_mat$prior_mean,
              dyn_covar_mat_col_cov = self$dyn_covariate_mat$prior_col_cov,
              dyn_cov_df = self$dyn_cov_mat$prior_df,
              dyn_cov_scale = self$dyn_cov_mat$prior_scale,
              meas_design_mat_const = self$meas_design_mat$constraint,
              meas_design_mat_mean_alt = self$meas_design_mat$prior_mean,
              meas_design_mat_cov_alt = self$meas_design_mat$prior_cov,
              meas_covar_mat_const = self$meas_covariate_mat$constraint,
              meas_covar_mat_mean_alt = self$meas_covariate_mat$prior_mean,
              meas_covar_mat_cov_alt = self$meas_covariate_mat$prior_cov,
              meas_cov_df = self$meas_cov_mat$prior_df,
              meas_cov_scale = self$meas_cov_mat$prior_scale,
              mh_rep = mh_rep,
              pg_rep = pg_rep,
              mh_adapt_start = mh_adapt_start,
              exact = exact,
              post_pred = post_pred,
              disp_prog = disp_prog
            )

            # Assign column names
            name_parts <- c(
              if (post_pred) self$data$get_data_name_vec(),
              self$data$get_latent_name_vec(),
              self$gp_name_vec,
              self$hyperparameter_names,
              if (!exact) self$dyn_design_mat$get_par_names(),
              self$dyn_covariate_mat$get_par_names(),
              self$dyn_cov_mat$get_par_names(),
              self$meas_design_mat$get_par_names(),
              self$meas_covariate_mat$get_par_names(),
              self$meas_cov_mat$get_par_names(), 
              "log_lik"
            )
            colnames(chain_sample) <- name_parts

            chain_sample
          }
        )
      }

      sample_draws <- posterior::as_draws(samples)

      self$posterior_samples <- sample_draws[(warm_up + 1):(iter + warm_up), , ]

      return(invisible(self))
    },
    get_samples = function(prior = FALSE) {
      if (prior) {
        self$prior_samples
      } else {
        self$posterior_samples
      }
    },
    plot_pred = function(prior = FALSE,
                         latent_smooth = FALSE, ci = 0.95,
                         true_latent = FALSE) {
      if (!require(ggplot2)) {
        stop("'ggplot2' required for plotting.")
      }

      n_indicators <- self$data$get_d_obs()
      data_name <- self$data$data_name
      latent_name <- self$data$latent_name
      n_time <- self$data$get_time()
      alpha <- (1 - ci) / 2
      time_index <- seq_len(n_time)

      ## Access samples without coppying
      plot_df_all <- data.frame()
      obs_df_all <- data.frame()
      true_df_all <- data.frame()

      for (i in seq_len(n_indicators)) {
        ## ---- Indicator samples ----
        indicator_names <- paste0("^", data_name, "\\[", i, ",")
        var_names <- grep(indicator_names, 
        posterior::variables(self$get_samples(prior)), 
        value = TRUE)

        samples_indicator <- posterior::as_draws_matrix(
          posterior::subset_draws(self$get_samples(prior), variable = var_names)
        )

        mean_vals <- colMeans(samples_indicator)
        lb_vals <- apply(samples_indicator, 2, quantile, alpha)
        ub_vals <- apply(samples_indicator, 2, quantile, 1 - alpha)

        indicator_df <- data.frame(
          time = time_index,
          mean = mean_vals,
          lb = lb_vals,
          ub = ub_vals,
          indicator = paste("Indicator", i),
          type = "indicator"
        )

        plot_df <- indicator_df

        ## Extract latent smooth if desired
        if (latent_smooth) {
          latent_names <- paste0("^", latent_name, "\\[", latent_smooth, ",")
          var_names <- grep(latent_names, 
          posterior::variables(self$get_samples(prior)), 
          value = TRUE)

          samples_latent <- posterior::as_draws_matrix(
            posterior::subset_draws(self$get_samples(prior), 
            variable = var_names)
          )

          latent_df <- data.frame(
            time = time_index,
            mean = colMeans(samples_latent),
            lb = apply(samples_latent, 2, quantile, alpha),
            ub = apply(samples_latent, 2, quantile, 1 - alpha),
            indicator = paste("Indicator", i),
            type = "latent"
          )

          plot_df <- rbind(plot_df, latent_df)
        }

        plot_df_all <- rbind(plot_df_all, plot_df)

        ## Extract observed data
        obs_df <- data.frame(
          time = time_index,
          value = self$data$data[i, ],
          indicator = paste("Indicator", i),
          type = "observed"
        )

        obs_df_all <- rbind(obs_df_all, obs_df)

        ## add true latent variable if desires
        if (!identical(true_latent, FALSE)) {
          true_df <- data.frame(
            time = time_index,
            value = true_latent,
            indicator = paste("Indicator", i),
            type = "true"
          )

          true_df_all <- rbind(true_df_all, true_df)
        }
      }

      ## Build ggplot
      p <- ggplot2::ggplot()

      indicator_df <- plot_df_all[plot_df_all$type == "indicator", ]

      p <- p +
        ggplot2::geom_ribbon(
          data = indicator_df,
          aes(x = time, ymin = lb, ymax = ub, fill = "Credible Interval"),
          alpha = 0.15
        )

      if (latent_smooth) {
        latent_df <- plot_df_all[plot_df_all$type == "latent", ]
        p <- p +
          ggplot2::geom_ribbon(
            data = latent_df,
            aes(x = time, ymin = lb, ymax = ub, fill = "Latent Smooth CI"),
            alpha = 0.25
          )
      }

      p <- p +
        ggplot2::geom_line(
          data = indicator_df,
          aes(x = time, y = mean, linetype = "Posterior Mean"),
          linewidth = 1
        ) +
        ggplot2::geom_point(
          data = obs_df_all,
          aes(x = time, y = value, shape = "Observed Data"),
          size = 1.8
        )

      if (!identical(true_latent, FALSE)) {
        p <- p +
          ggplot2::geom_line(
            data = true_df_all,
            aes(x = time, y = value, linetype = "True Latent"),
            linewidth = 1
          )
      }

      p <- p +
        ggplot2::facet_wrap(~indicator, scales = "free_y") +
        ggplot2::scale_fill_manual(
          name = "",
          values = c(
            "Credible Interval" = "#1B4F72",
            "Latent Smooth CI"  = "#1B4F72"
          )
        ) +
        ggplot2::scale_linetype_manual(
          name = "",
          values = c(
            "Posterior Mean" = "dashed",
            "True Latent" = "solid"
          )
        ) +
        ggplot2::scale_shape_manual(
          name = "",
          values = c("Observed Data" = 16)
        ) +
        ggplot2::labs(
          x = "Time",
          y = "Latent Variable",
          title = if (prior) "Prior Predictive" else "Posterior Predictive"
        ) +
        ggplot2::theme_minimal(base_size = 14) +
        ggplot2::theme(
          legend.position = "bottom",
          legend.box = "horizontal",
          strip.text = ggplot2::element_text(size = 12, face = "bold"),
          plot.title = ggplot2::element_text(size = 16, face = "bold"),
          axis.line = ggplot2::element_line(color = "black"),
          axis.ticks = ggplot2::element_line(color = "black"),
          panel.grid.minor = ggplot2::element_blank()
        )

      print(p)
      return(p)
    }
  )
)

# Would probably be good to have these as seperate classes at some point !!!
gpssm_gp <- function(
    dprior, rprior, hyperparameter_names,
    m_basis = NULL, c = NULL, S = NULL, gp_name = "gp") {
  if (is.null(m_basis) & is.null(c) & is.null(S)) {
    exact <- TRUE
  } else if (!is.null(m_basis) & !is.null(c) & !is.null(S)) {
    exact <- FALSE
  } else {
    stop("GP specification is not consistent. Please either provide
     input for an exact or a Hilbert-space approximate GP.")
  }

  if (exact) {
    m_basis_list <- lapply(m_basis, function(x) 0)
    basis_fun_index <- do.call(expand.grid, m_basis_list)
    c <- 0
    S <- 0
  } else {
    m_basis_list <- lapply(m_basis, function(x) 1:x)
    basis_fun_index <- do.call(expand.grid, m_basis_list)
  }

  ## Wrap rprior() to log(rprior())
  rprior_tans <- function() {
    return(log(rprior()))
  }
  # Wrap dprior() to log-density under exp-transform + Jacobian
  dprior_trans <- function(log_pars) {
    return(dprior(exp(log_pars), log = TRUE) + sum(log_pars))
  }

  list(
    basis_fun_index = as.matrix(basis_fun_index),
    boundry_factor = c * S,
    hyperparameter_names = hyperparameter_names,
    gp_name = gp_name,
    dprior = dprior_trans,
    rprior = rprior_tans,
    gp_exact = exact
  )
}

gpss_model_part <- function(design_mat, covariate_mat = NULL, cov_mat) {
  if (is.null(covariate_mat)) {
    if ("gpssm_design_mat" %in% class(design_mat)) {
      if (!design_mat$uses_alt_prior()) {
        covariate_mat <- gpssm_design_mat$new(
          name = paste0(design_mat$name, "_covar"),
          constraint = matrix(numeric(0), nrow(design_mat$constraint), 0),
          prior_mean = matrix(numeric(0), nrow(design_mat$constraint), 0),
          prior_col_cov = matrix(numeric(0), 0, 0)
        )
      } else {
        covariate_mat <- gpssm_design_mat$new(
          name = paste0(design_mat$name, "_covar"),
          constraint = matrix(numeric(0), nrow(design_mat$constraint), 0),
          prior_mean_alt = numeric(0),
          prior_cov_alt = matrix(numeric(0), 0, 0)
        )
      }
    } else if ("gpssm_mn_prior" %in% class(design_mat)) {
      covariate_mat <- gpssm_mn_prior$new(
        name = paste0(design_mat$name, "_covariate"),
        prior_mean = matrix(numeric(0), nrow(design_mat$prior_mean), 0),
        prior_col_cov = matrix(numeric(0), 0, 0)
      )
    } else if ("gpssm_mvn_prior" %in% class(design_mat)) {
      covariate_mat <- gpssm_mvn_prior$new(
        name = paste0(design_mat$name, "_covariate"),
        constraint = matrix(numeric(0), nrow(design_mat$constraint), 0),
        prior_mean = numeric(0),
        prior_cov = matrix(numeric(0), 0, 0)
      )
    }
  }

  list(
    design_mat = design_mat,
    covariate_mat = covariate_mat,
    cov_mat = cov_mat
  )
}
