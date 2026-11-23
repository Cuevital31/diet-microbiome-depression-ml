# Firth penalised logistic regression (logistf) for a list of model specifications.
# Usage: Rscript firth_logistf.R <data.csv> <spec.csv> <out.csv>
#   data.csv : column y (0/1), z-scored feature columns, confounders Age, BMI, SQ_all, PA (original scale)
#   spec.csv : fit_id, feature (column name), covariates ("" or names separated by "+")
# Profile penalised-likelihood 95% CI and penalised likelihood-ratio p-value (pl = TRUE).
suppressPackageStartupMessages(library(logistf))
args <- commandArgs(trailingOnly = TRUE)
dat <- read.csv(args[1], check.names = FALSE)
spec <- read.csv(args[2], stringsAsFactors = FALSE, check.names = FALSE)
spec$covariates[is.na(spec$covariates)] <- ""
rows <- vector("list", nrow(spec))
for (i in seq_len(nrow(spec))) {
  f <- spec$feature[i]
  covs <- if (nchar(spec$covariates[i]) > 0) strsplit(spec$covariates[i], "+", fixed = TRUE)[[1]] else character(0)
  d <- data.frame(y = dat$y, x = dat[[f]])
  for (cv in covs) d[[cv]] <- dat[[cv]]
  form <- as.formula(paste("y ~", paste(c("x", covs), collapse = " + ")))
  fit <- logistf(form, data = d, pl = TRUE, alpha = 0.05,
                 control = logistf.control(maxit = 100), plcontrol = logistpl.control(maxit = 100))
  rows[[i]] <- data.frame(
    fit_id = spec$fit_id[i], beta = unname(fit$coefficients["x"]), ci_low = unname(fit$ci.lower["x"]),
    ci_high = unname(fit$ci.upper["x"]), p = unname(fit$prob["x"]),
    converged = all(as.numeric(fit$iter) < 100), stringsAsFactors = FALSE)
}
write.csv(do.call(rbind, rows), args[3], row.names = FALSE)
cat("logistf", as.character(packageVersion("logistf")), "from", find.package("logistf"), "|", R.version.string,
    "| fits:", length(rows), "\n")
