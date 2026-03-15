is_psd_chol <- function(psd_matrix) {
  if (!isSymmetric(psd_matrix)) {
    return(FALSE)
  }
  res <- tryCatch(chol(psd_matrix), error = function(e) NULL)
  return(!is.null(res))
}
vec2mat <- function(vector, mask) {
  out <- mask
  ind <- which(!is.finite(mask))
  out[ind] <- vector
  out
}
add_error <- function(msg) {
  errors <<- c(errors, msg)
}
