## gpssmR: Gaussian Process State-Space Modelling in R

### Status

*Development version*: This package is a work in progress. 
There is **no guarantee** of correctness or stability of any results
that were obtained using it.

### Synopsis

`gpssmR` provides tools for inferring Gaussian Process State-Space Models
(GPSSMs) in R. It includes a Bayesian sampler for both exact and Hilbert-space
approximate GPSSMs. The sampler is implemented in C++ and can be accessed from
R via Rcpp. The package also provides R classes for defining GPSSMs and for
accessing the samplers for their posterior and prior predictive distributions.

### Structure

- `gpssm.R`: Core R classes and user-facing interface
- C++ backend: Sampler implementation exposed via `Rcpp`

### Installation

Install from source:

```r
# install.packages("devtools")
devtools::install_github("Jan-Ian-Failenschmid/gpssmR")

```