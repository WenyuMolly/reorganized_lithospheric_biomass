# Depth_and_Temperature_GLM_wenyu.R
# -------------------------------------------------------------------
# GLMnet (elastic net) fit for log10 cell density using features:
#   - Depth (log10 meters) and
#   - Temperature (°C) computed from site MAST and geothermal gradient.
#
# Then integrate predicted density along depth for each grid cell using
# explicit slice thickness dz, aggregate into user-defined depth bins,
# and multiply by horizontal area to get total cells.
#
# Units sanity:
#   prediction: cells / cm^3
#   line integral (sum * dz_cm): cells / cm^2
#   multiply by area_cm2: cells
# -------------------------------------------------------------------

library(doParallel)
library(foreach)
library(glmnet)
library(fields)
library(nlstools)

registerDoParallel(cores = 15)

script_dir <- dirname(normalizePath(sys.frame(1)$ofile))
project_root <- normalizePath(file.path(script_dir, "../../.."))
input_dir <- file.path(project_root, "data/processed/continental/modified_magnabosco")
output_dir <- file.path(project_root, "runs/continental/latest/modified_magnabosco")
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
setwd(output_dir)

# -------------------------------
# 1) Load and format data
# -------------------------------
gridCells <- read.csv(file.path(input_dir, "metadata_with_merged_depth_and_gradient.csv"), stringsAsFactors = FALSE)

GreenlandFID <- c(3774:3776,3759:3762,3737:3742,3711:3715,3681:3685,3639:3643,3584:3588,3522:3525,3452:3454,3378:3380)
AntarcFID    <- 3791:4163

# Depth-bin edges in km (will auto-extend to max z if needed)
depth_edges_km <- c(0, 0.3, 0.7, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10)

# Convert to meters for the integration grid; clamp lower bound to 1 m
depth_edges_m <- pmax(1, depth_edges_km * 1000)
nbins <- length(depth_edges_m) - 1

df.trimmed <- read.csv(file.path(input_dir, "cores_with_gradient_filled.csv"))
all <- df.trimmed[df.trimmed$MethodCM == "direct", ]
all$Depth    <- as.numeric(all$Depth)       # meters
all$cellsPer <- as.numeric(all$cellsPer)    # cells / cm^3

# -------------------------------
# 2) Bootstrap index + bin edges update
# -------------------------------
myIndices   <- as.matrix(read.csv(file.path(input_dir, "1000_indices_for_bootstrap.csv"), header = FALSE))
bootstraps  <- nrow(myIndices)

depthsToIterate <- gridCells$maxdepth * 1000  # meters
max_z_km <- max(depthsToIterate, na.rm = TRUE) / 1000
if (is.finite(max_z_km) && max_z_km > max(depth_edges_km)) {
  depth_edges_km <- c(depth_edges_km, max_z_km)
  depth_edges_m  <- pmax(1, depth_edges_km * 1000)
  nbins <- length(depth_edges_m) - 1
}

# -------------------------------
# 3) Output containers
# -------------------------------
cv.byDepthAll        <- matrix(NA_real_, nrow = bootstraps, ncol = nbins)  # cells per bin (global sum)
temperature.error    <- numeric(bootstraps)
cv.error             <- numeric(bootstraps)
cv.rsq               <- numeric(bootstraps)

glm_params <- data.frame(
  bootstrap        = 1:bootstraps,
  lambda_min       = NA_real_,
  coef_Intercept   = NA_real_,
  coef_Depth       = NA_real_,
  coef_temperature = NA_real_
)

cv.biomass          <- numeric(bootstraps)  # global total cells per bootstrap
cv.byGridResult     <- data.frame(matrix(NA_real_, nrow = nrow(gridCells), ncol = bootstraps))

# -------------------------------
# 4) Begin bootstrap loop
# -------------------------------
ptm <- proc.time()

for (n in 1:bootstraps) {
  message(sprintf("Bootstrap iteration: %d / %d", n, bootstraps))
  train_ind <- myIndices[n, ]

  # --- Gradient outlier clipping (1%–99%) on both cores and grids
  q1  <- quantile(all$gradient,       0.01, na.rm = TRUE)
  q99 <- quantile(all$gradient,       0.99, na.rm = TRUE)
  g1  <- quantile(gridCells$gradient, 0.01, na.rm = TRUE)
  g99 <- quantile(gridCells$gradient, 0.99, na.rm = TRUE)

  all$gradient[all$gradient < q1]  <- q1
  all$gradient[all$gradient > q99] <- q99
  gridCells$gradient[gridCells$gradient < g1]  <- g1
  gridCells$gradient[gridCells$gradient > g99] <- g99

  # --- Temperature in cores (°C): mast + (Depth[m] -> km) * gradient[°C/km]
  all$temperature <- all$mast + all$Depth * all$gradient / 1000

  # --- GLMnet training set (log10 transform on response and depth)
  keep      <- c("cellsPer", "Depth", "temperature")
  train     <- all[train_ind, colnames(all) %in% keep]
  test      <- all[-train_ind, colnames(all) %in% keep]

  train$Depth    <- log10(pmax(train$Depth, 1))  # avoid log10(0)
  test$Depth     <- log10(pmax(test$Depth,  1))
  train$cellsPer <- log10(pmax(train$cellsPer, 1e-12))
  test$cellsPer  <- log10(pmax(test$cellsPer,  1e-12))

  train <- train[complete.cases(train), ]
  test  <- test[complete.cases(test),  ]

  train.x <- model.matrix(cellsPer ~ ., train)[, -1]
  train.y <- train$cellsPer
  test.x  <- model.matrix(cellsPer ~ ., test)[, -1]
  test.y  <- test$cellsPer

  glmfit <- glmnet(train.x, train.y)
  cv.fit <- cv.glmnet(train.x, train.y)

  coef_min <- as.matrix(coef(cv.fit, s = "lambda.min"))
  get_coef <- function(name) if (name %in% rownames(coef_min)) as.numeric(coef_min[name, 1]) else 0

  glm_params$lambda_min[n]       <- cv.fit$lambda.min
  glm_params$coef_Intercept[n]   <- get_coef("(Intercept)")
  glm_params$coef_Depth[n]       <- get_coef("Depth")
  glm_params$coef_temperature[n] <- get_coef("temperature")

  cv.pred  <- as.numeric(predict(cv.fit, newx = test.x, s = "lambda.min"))
  ss_res   <- sum((test.y - cv.pred)^2)
  ss_tot   <- sum((test.y - mean(test.y))^2)
  cv.rsq[n] <- 1 - ss_res/ss_tot
  cv.error[n] <- mean((cv.pred - test.y)^2)

  # -------------------------------
  # 5) Numerical integration with explicit dz
  # -------------------------------
  # Horizontal area (m^2 -> cm^2)
  A_cm2 <- gridCells$grid_area_m2 * 1e4

  # Integration step: dz in meters (choose as needed)
  dz_m  <- 0.01         # 0.01 m = 1 cm
  dz_cm <- dz_m * 100   # m -> cm

  combiner <- function(acc, value) {
    acc$total <- c(acc$total, value$total)            # cells/cm^2 per grid
    if (is.null(acc$bybin)) acc$bybin <- value$bybin  # [grid x nbins], cells/cm^2
    else acc$bybin <- rbind(acc$bybin, value$bybin)
    acc
  }

  res <- foreach(i = 1:length(depthsToIterate),
                 .combine = combiner,
                 .init = list(total = numeric(0), bybin = NULL),
                 .export = c("gridCells", "depth_edges_m", "nbins", "train.x",
                             "cv.fit", "GreenlandFID", "AntarcFID", "dz_m", "dz_cm"),
                 .packages = c("glmnet")) %dopar% {

    zmax_m <- depthsToIterate[i]
    if (!is.finite(zmax_m) || zmax_m <= 0) {
      return(list(total = 0, bybin = matrix(0, nrow = 1, ncol = nbins)))
    }

    # Depth slices from surface (0 m) to zmax (inclusive) with step dz_m
    mySlices <- seq(0, zmax_m, by = dz_m)

    # Features for prediction
    patch <- data.frame(gridCells[i, ])
    patch <- patch[rep(1, length(mySlices)), ]
    depth_for_log10 <- pmax(mySlices, 1)  # only for the feature
    patch$Depth <- log10(depth_for_log10)

    if (patch$FID[1] %in% c(GreenlandFID, AntarcFID)) {
      patch$temperature <- 0 + mySlices * gridCells[i, ]$gradient / 1000  # °C
    } else {
      patch$temperature <- gridCells[i, ]$MEAN_Annual_Temp + mySlices * gridCells[i, ]$gradient / 1000
    }

    patch$dummy <- 1
    keep <- c("cellsPer", "Depth", "temperature")
    patch <- patch[, colnames(patch) %in% c(keep, "dummy")]
    patch <- model.matrix(dummy ~ ., patch)

    # Align columns with training matrix
    needed <- colnames(train.x)
    have   <- colnames(patch)

    extra <- setdiff(have, needed)
    if (length(extra) > 0) patch <- patch[, setdiff(have, extra), drop = FALSE]

    miss <- setdiff(needed, colnames(patch))
    if (length(miss) > 0) {
      patch <- cbind(
        patch,
        matrix(0, nrow = nrow(patch), ncol = length(miss), dimnames = list(NULL, miss))
      )
    }
    patch <- patch[, needed, drop = FALSE]

    # Predict log10 density -> linear density (cells/cm^3)
    preds_cm3 <- as.numeric(10 ^ predict(cv.fit, newx = data.matrix(patch), s = "lambda.min"))

    # Multiply by dz (cm) to get line integral (cells/cm^2) per slice
    preds_line <- preds_cm3 * dz_cm

    # Bin by depth (meters); use left-closed/right-open to avoid double-count at edges
    bin_idx <- cut(mySlices,
                   breaks = depth_edges_m,
                   right = FALSE, include.lowest = TRUE)
    g <- as.integer(bin_idx)
    valid <- !is.na(g)

    sums_full <- rep(0, nbins)
    if (any(valid)) {
      s <- rowsum(preds_line[valid], g[valid], reorder = FALSE)  # cells/cm^2 per bin
      sums_full[as.integer(rownames(s))] <- as.numeric(s[, 1])
    }

    list(
      total = sum(preds_line),                # cells/cm^2 (integrated along z)
      bybin = matrix(sums_full, nrow = 1)     # [1 x nbins], cells/cm^2
    )
  }

  # Convert line integrals to total cells by multiplying horizontal area
  results_line <- res$total                       # cells/cm^2 per grid
  totals_cells <- results_line * A_cm2            # cells per grid
  cv.byGridResult[, n] <- totals_cells
  cv.biomass[n]        <- sum(totals_cells)       # global total cells

  # Per-bin global totals: multiply each grid row by its area, then column-sum
  bybin_line <- res$bybin                          # [ngrids x nbins], cells/cm^2
  bybin_cells <- sweep(bybin_line, 1, A_cm2, "*")  # per-grid area -> cells
  cv.byDepthAll[n, ] <- colSums(bybin_cells)       # global cells per bin
}

proc.time() - ptm

# -------------------------------
# 6) Write outputs
# -------------------------------
write.table(cv.biomass,
            file = "glm_depthtemp122_cv.biomass.csv",
            sep = ",", row.names = FALSE, col.names = FALSE)

write.table(cv.error,
            file = "glm_depthtemp122_cv.error.csv",
            sep = ",", row.names = FALSE, col.names = FALSE)

write.table(cv.byGridResult,
            file = "glm_depthtemp122_cvGridResult.csv",
            sep = ",", row.names = FALSE, col.names = FALSE)

write.table(cv.rsq,
            file = "glm_depthtemp122_cv.rsq.csv",
            sep = ",", row.names = FALSE, col.names = FALSE)

write.csv(glm_params,
          file = "glm_depthtemp122_cv.parameters.csv",
          row.names = FALSE)

# Matrix of per-bin totals by bootstrap
by_depth_matrix <- data.frame(
  depth_top_km = depth_edges_km[-length(depth_edges_km)],
  depth_bot_km = depth_edges_km[-1]
)
by_depth_matrix <- cbind(by_depth_matrix, as.data.frame(t(cv.byDepthAll)))
colnames(by_depth_matrix)[-(1:2)] <- paste0("iter_", seq_len(bootstraps))
write.csv(by_depth_matrix, "glm_depthtemp122_cv_biomass_by_depth_matrix.csv", row.names = FALSE)

# Summary stats per bin across bootstraps (mean + median + 95% CI)
byDepth_mean  <- apply(cv.byDepthAll, 2, mean,     na.rm = TRUE)
byDepth_med   <- apply(cv.byDepthAll, 2, median,   na.rm = TRUE)
byDepth_lo95  <- apply(cv.byDepthAll, 2, quantile, probs = 0.025, na.rm = TRUE)
byDepth_hi95  <- apply(cv.byDepthAll, 2, quantile, probs = 0.975, na.rm = TRUE)

write.csv(
  data.frame(
    depth_top_km = depth_edges_km[-length(depth_edges_km)],
    depth_bot_km = depth_edges_km[-1],
    total_mean   = byDepth_mean,
    total_median = byDepth_med,
    total_lo95   = byDepth_lo95,
    total_hi95   = byDepth_hi95
  ),
  "glm_depthtemp122_cv_biomass_by_depth_summary.csv",
  row.names = FALSE
)

message("DONE. Integration now multiplies by dz (cm) and area (cm^2); bins use left-closed/right-open intervals; bin stats include mean/median/95% CI.")
