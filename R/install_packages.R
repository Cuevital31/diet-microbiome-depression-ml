# Install the R package used for the Firth regressions into a separate library.
# Usage: Rscript R/install_packages.R <library folder>
# Reported results: R 4.5.1 with logistf 1.26.1 (and its CRAN dependencies).
args <- commandArgs(trailingOnly = TRUE)
lib <- if (length(args) >= 1) args[1] else "r-lib"
dir.create(lib, showWarnings = FALSE, recursive = TRUE)
.libPaths(c(normalizePath(lib), .libPaths()))
repos <- "https://cloud.r-project.org"
if (!requireNamespace("logistf", lib.loc = lib, quietly = TRUE)) {
  install.packages("logistf", lib = lib, repos = repos)
}
if (as.character(packageVersion("logistf", lib.loc = lib)) != "1.26.1") {
  # CRAN has moved on: install the archived source version (requires Rtools on Windows)
  remove.packages("logistf", lib = lib)
  install.packages(paste0(repos, "/src/contrib/Archive/logistf/logistf_1.26.1.tar.gz"), lib = lib,
                   repos = NULL, type = "source")
}
stopifnot(as.character(packageVersion("logistf", lib.loc = lib)) == "1.26.1")
cat("logistf", as.character(packageVersion("logistf", lib.loc = lib)), "installed in", normalizePath(lib), "|",
    R.version.string, "\n")
