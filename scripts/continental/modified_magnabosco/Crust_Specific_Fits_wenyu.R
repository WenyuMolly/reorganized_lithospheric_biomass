## ================================================================
## Regionally specific depth power fit (CSF)
## - parallelized bootstraps
## - depth-bin (layer) integration outputs
## - robust parameter layout across bootstraps
## ================================================================

options(stringsAsFactors = FALSE)

suppressPackageStartupMessages({
  library(doParallel)
  library(foreach)
  library(ggplot2)
})

## ---------- 0) basic path ----------
script_dir <- dirname(normalizePath(sys.frame(1)$ofile))
project_root <- normalizePath(file.path(script_dir, "../../.."))
input_dir <- file.path(project_root, "data/processed/continental/modified_magnabosco")
output_dir <- file.path(project_root, "runs/continental/latest/modified_magnabosco")
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
setwd(output_dir)

## ---------- 1) read grids and dataset ----------
gridCells <- read.csv(file.path(input_dir, "metadata_with_merged_depth_and_gradient.csv"), stringsAsFactors = FALSE)

## Greenland / Antarctica FID list
GreenlandFID <- c(3774:3776,3759:3762,3737:3742,3711:3715,3681:3685,3639:3643,3584:3588,3522:3525,3452:3454,3378:3380)
AntarcFID    <- 3791:4163

## factors
gridCells$rechargeType  <- gridCells$rechargeDepth
gridCells$combined      <- factor(paste(gridCells$rechargeVolume, gridCells$crustScheme2))
gridCells$rechargeVolume<- factor(gridCells$rechargeVolume)
gridCells$rechargeType  <- factor(gridCells$rechargeDepth)
gridCells$crustScheme2  <- factor(gridCells$crustScheme2)
gridCells$rechargeFull  <- factor(gridCells$Descriptio)
gridCells$combined_cv   <- factor(gridCells$combined_cv)
gridCells$combined_cr   <- factor(gridCells$combined_cr)
gridCells$rechargeShort <- factor(gridCells$rechargeShort)

crustLevels <- sort(unique(as.character(gridCells$crustScheme2)))

## ---------- 2) depth bins setting ----------
depth_edges_km <- c(0, 0.3, 0.7, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
depth_edges_m  <- pmax(1, depth_edges_km * 1000)
nbins          <- length(depth_edges_m) - 1

## ---------- 3) observed dataset ----------
df.trimmed <- read.csv(file.path(project_root, "data/raw/continental/cores_with_PCR.csv"))
all <- df.trimmed[df.trimmed$MethodCM == "direct", ]

all$Depth    <- suppressWarnings(as.numeric(all$Depth))     # m
all$cellsPer <- suppressWarnings(as.numeric(all$cellsPer))  # cells/cm^3
all <- subset(all, is.finite(Depth) & is.finite(cellsPer) & Depth > 0 & cellsPer > 0)

## train log cols
all$d <- log10(all$Depth)
all$c <- log10(all$cellsPer)

if (!"crustScheme2" %in% names(all)) {
  stop("Column 'crustScheme2' not found in 'all'. Please ensure it's present.")
}

## Bootstraps index
myIndices   <- as.matrix(read.csv(file.path(input_dir, "1000_indices_for_bootstrap.csv"), header = FALSE))
bootstraps  <- nrow(myIndices)

depthsToIterate <- gridCells$maxdepth * 1000

max_z_km <- max(depthsToIterate, na.rm = TRUE) / 1000
if (is.finite(max_z_km) && max_z_km > max(depth_edges_km)) {
  depth_edges_km <- c(depth_edges_km, max_z_km)
  depth_edges_m  <- pmax(1, depth_edges_km * 1000)
  nbins          <- length(depth_edges_m) - 1
}

## ---------- 4) parallel settings ----------
cores_to_use <- max(1L, parallel::detectCores(logical = TRUE) - 1L)
registerDoParallel(cores = cores_to_use)
message(sprintf("Registered %d workers for parallel bootstraps.", cores_to_use))

## ---------- 5) linear model（crust specific + per-bin integration） ----------
linearModel <- function(var, map, train, test, depth_edges_m) {

  var_chr <- as.character(var)
  map_chr <- as.character(map)

  biomass_total <- 0
  error_mat     <- matrix(nrow = 0, ncol = 2)
  univar        <- sort(unique(var_chr))

  ## [a,b,R2] × length(crustLevels)
  loopAB_full <- rep(NA_real_, 3 * length(crustLevels))
  gridVec     <- rep(NA_real_, nrow(gridCells))
  byDepthVec  <- rep(0, length(depth_edges_m) - 1)

  for (uv in univar) {
    sel_tr <- which(var_chr == uv)

    sel_ts <- which(as.character(test$crustScheme2) == uv)

    pf.train <- train[sel_tr, , drop = FALSE]
    pf.test  <- test[sel_ts,  , drop = FALSE]

    ## lm: log10(cells) ~ log10(depth)
    powerFit <- lm(c ~ d, data = pf.train)
    a <- unname(coef(powerFit)[1])
    b <- unname(coef(powerFit)[2])
    rsq <- summary(powerFit)$r.squared

    pos <- match(uv, crustLevels)
    if (!is.na(pos)) {
      base <- (pos - 1L) * 3L
      loopAB_full[base + 1L] <- a
      loopAB_full[base + 2L] <- b
      loopAB_full[base + 3L] <- rsq
    }

    if (nrow(pf.test) > 0) {
      powerResult <- as.numeric(predict(powerFit, newdata = pf.test))
      error_mat   <- rbind(error_mat, cbind(pf.test$c, powerResult))
    }

    idx_g <- which(map_chr == uv)
    for (g in idx_g) {
      A_cm2 <- gridCells$grid_area_m2[g] * 100 * 100   # m^2 → cm^2
      zmax  <- depthsToIterate[g]                      # m
      if (is.na(zmax) || zmax < 1) { gridVec[g] <- 0; next }

      ## （cells/cm^3 as f(depth[m])）
      if (gridCells$FID[g] %in% GreenlandFID) {
        surface_term <- 15000000000
        integralFun  <- function(x) { (10^7.73) * x^(-0.66) }
      } else if (gridCells$FID[g] %in% AntarcFID) {
        surface_term <- 2150000000
        integralFun  <- function(x) { (10^6) * x^(-0.66) }
      } else {
        surface_term <- 0
        integralFun  <- function(x) { (10^a) * x^b }
      }

      grid_sum_all <- 0
      for (k in 1:(length(depth_edges_m) - 1)) {
        z1 <- depth_edges_m[k]
        z2 <- depth_edges_m[k + 1]
        if (z1 >= zmax) break
        up <- min(z2, zmax)

        ## ∫(cells/cm^3) dx(m)，* 100 : m → cm；and * A_cm2 → cells
        contrib_cells_cm3_m <- if (up > z1) integrate(integralFun, z1, up, rel.tol = 1e-6)$value else 0
        contrib_total       <- A_cm2 * contrib_cells_cm3_m * 100

        byDepthVec[k] <- byDepthVec[k] + contrib_total
        grid_sum_all  <- grid_sum_all  + contrib_total

        ## Add surface term to the first layers in polar regions
        if (k == 1 && surface_term > 0) {
          add0 <- A_cm2 * surface_term * 100
          byDepthVec[k] <- byDepthVec[k] + add0
          grid_sum_all  <- grid_sum_all  + add0
        }
      }

      gridVec[g]    <- grid_sum_all
      biomass_total <- biomass_total + grid_sum_all
    }
  }

  ## MSE
  mse <- if (nrow(error_mat) > 0) mean((error_mat[,1] - error_mat[,2])^2, na.rm = TRUE) else NA_real_
  biomassAndError <- c(biomass_total, mse)

  if (is.finite(mse) && (mse > 2 || mse == 0)) {
    list(estimate = rep(NA_real_, 2),
         grid     = rep(NA_real_, nrow(gridCells)),
         params   = rep(NA_real_, 3 * length(crustLevels)),
         by_depth = rep(NA_real_, length(depth_edges_m) - 1))
  } else {
    list(estimate = biomassAndError,
         grid     = gridVec,
         params   = loopAB_full,
         by_depth = byDepthVec)
  }
}

## ---------- 6) parallel over bootstraps ----------
ptm <- proc.time()

res_list <- foreach(n = 1:bootstraps,
                    .packages = c("stats"),
                    .export   = c("gridCells","depthsToIterate",
                                  "GreenlandFID","AntarcFID",
                                  "depth_edges_m","linearModel",
                                  "crustLevels")) %dopar% {
  if (n %% 50 == 0) message(sprintf("Bootstrap %d / %d", n, bootstraps))

  trainSet <- all[myIndices[n,], , drop = FALSE]
  testSet  <- all[-myIndices[n,], , drop = FALSE]

  if (!"crustScheme2" %in% names(testSet)) {
    testSet$crustScheme2 <- all$crustScheme2[-myIndices[n,]]
  }

  if (sum(table(trainSet$crustScheme2) < 4) > 0) {
    list(estimate = c(NA_real_, NA_real_),
         grid     = rep(NA_real_, nrow(gridCells)),
         params   = rep(NA_real_, 3 * length(crustLevels)),
         by_depth = rep(NA_real_, nbins))
  } else {
    linearModel(trainSet$crustScheme2, gridCells$crustScheme2,
                trainSet, testSet, depth_edges_m)
  }
}

elapsed <- proc.time() - ptm
message(sprintf("Bootstraps finished in %.2f sec", elapsed[3]))

## ---------- 7) gather results ----------
newEstimate <- do.call(rbind, lapply(res_list, `[[`, "estimate"))   # B x 2  (total, mse)
gridValues  <- do.call(cbind, lapply(res_list, `[[`, "grid"))       # Ngrid x B
parameters  <- do.call(rbind, lapply(res_list, `[[`, "params"))     # B x (3*K)
byDepthAll  <- do.call(rbind, lapply(res_list, `[[`, "by_depth"))   # B x nbins

## ---------- 8) plots ----------
crustTypes    <- crustLevels                   
median_params <- apply(parameters, 2, median, na.rm = TRUE)
cols          <- grDevices::rainbow(length(crustTypes))

par(mfrow = c(3, 2))
for (i in seq_along(crustTypes)) {
  tmp <- all[all$crustScheme2 == crustTypes[i], ]
  if (nrow(tmp) == 0) next
  plot(tmp$d, tmp$c, col = cols[i], xlim = c(-1, 4), ylim = c(3, 10),
       main = sprintf("%s; median R² = %s", crustTypes[i],
                      toString(round(median_params[(i-1)*3+3], 2))))
  ## data=tmp
  tmp.lm <- lm(c ~ d, data = tmp)
  newd  <- data.frame(d = tmp$d)
  pr    <- predict(tmp.lm, newd, interval = "predict")
  abline(a = median_params[(i-1)*3+1], b = median_params[(i-1)*3+2], col = cols[i])
  lines(tmp$d, pr[,2])
  lines(tmp$d, pr[,3])
  print(mean(pr[,3] - pr[,2], na.rm = TRUE))
}

totdat.lm <- lm(c ~ d, data = all)
newd_all  <- data.frame(d = all$d)
pr_all    <- predict(totdat.lm, newd_all, interval = "predict")
plot(all$d, all$c, main = sprintf("Total Dataset; R² = %.2f", summary(totdat.lm)$r.squared))
abline(a = coef(totdat.lm)[1], b = coef(totdat.lm)[2], col = "gray")
lines(all$d, pr_all[,2], col = "gray")
lines(all$d, pr_all[,3], col = "gray")
print(mean(pr_all[,3] - pr_all[,2], na.rm = TRUE))

## ---------- 9) export ----------
write.csv(newEstimate, "CSF_bootstrap_total_biomass_and_mse.csv", row.names = FALSE)
write.csv(gridValues,  "CSF_bootstrap_grid_cell_biomass.csv",     row.names = FALSE)
write.csv(parameters,  "CSF_bootstrap_model_parameters.csv",      row.names = FALSE)

## R^2
if (ncol(parameters) >= 3) {
  r2_columns <- seq(3, ncol(parameters), by = 3)
  r2_matrix  <- parameters[, r2_columns, drop = FALSE]
  write.csv(r2_matrix, "CSF_bootstrap_crusttype_rsquare.csv", row.names = FALSE)
} else {
  r2_matrix <- NULL
}

## .RData save
save(newEstimate, gridValues, parameters, r2_matrix, file = "CSF_full_bootstrap_results_with_rsquare.RData")

byDepth_wide <- data.frame(
  depth_top_km = depth_edges_km[-length(depth_edges_km)],
  depth_bot_km = depth_edges_km[-1]
)
byDepth_wide <- cbind(byDepth_wide, as.data.frame(t(byDepthAll)))   # iter_1..B
colnames(byDepth_wide)[-(1:2)] <- paste0("iter_", seq_len(bootstraps))
write.csv(byDepth_wide, "CSF_bootstrap_biomass_by_depth_matrix.csv", row.names = FALSE)

byDepth_mean   <- apply(byDepthAll, 2, mean,     na.rm = TRUE)
byDepth_median <- apply(byDepthAll, 2, median,   na.rm = TRUE)
byDepth_lo     <- apply(byDepthAll, 2, quantile, probs = 0.025, na.rm = TRUE)
byDepth_hi     <- apply(byDepthAll, 2, quantile, probs = 0.975, na.rm = TRUE)

byDepth_summary <- data.frame(
  depth_top_km = depth_edges_km[-length(depth_edges_km)],
  depth_bot_km = depth_edges_km[-1],
  total_mean   = byDepth_mean,
  total_median = byDepth_median,
  total_lo95   = byDepth_lo,
  total_hi95   = byDepth_hi
)
write.csv(byDepth_summary, "CSF_bootstrap_biomass_by_depth_summary.csv", row.names = FALSE)

message("All done.")
