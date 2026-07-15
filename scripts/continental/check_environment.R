#!/usr/bin/env Rscript

# Verify the R runtime required by both continental biomass workflows.
required_packages <- c(
  "foreach",
  "doParallel",
  "glmnet",
  "fields",
  "nlstools",
  "ggplot2"
)

missing_packages <- required_packages[
  !vapply(required_packages, requireNamespace, logical(1), quietly = TRUE)
]

if (length(missing_packages) > 0) {
  cat("Missing required R packages:\n")
  cat(paste0("- ", missing_packages, collapse = "\n"), "\n")
  quit(status = 1)
}

cat(R.version.string, "\n")
package_versions <- vapply(
  required_packages,
  function(package_name) as.character(packageVersion(package_name)),
  character(1)
)
print(data.frame(package = required_packages, version = package_versions, row.names = NULL))
