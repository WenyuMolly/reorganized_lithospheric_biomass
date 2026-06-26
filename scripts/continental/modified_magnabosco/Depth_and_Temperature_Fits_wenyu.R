script_dir <- dirname(normalizePath(sys.frame(1)$ofile))
project_root <- normalizePath(file.path(script_dir, "../../.."))
input_dir <- file.path(project_root, "data/processed/continental/modified_magnabosco")
output_dir <- file.path(project_root, "runs/continental/latest/modified_magnabosco")
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
setwd(output_dir)

## Depth and Temperature Fits

gridCells = read.csv(file.path(input_dir, "metadata_with_merged_depth_and_gradient.csv"),stringsAsFactors = FALSE)
GreenlandFID = c(3774:3776,3759:3762,3737:3742,3711:3715,3681:3685,3639:3643,3584:3588,3522:3525,3452:3454,3378:3380)
AntarcFID = 3791:4163 

# -------- Depth bin settings (km) --------
# depth_edges_km <- c(0, 0.5, 2, 5, 10, 20, 35)
depth_edges_km <- c(0, 0.3, 0.7, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10)

depth_edges_m <- pmax(1, depth_edges_km * 1000)
nbins <- length(depth_edges_m) - 1

df.trimmed = read.csv(file.path(input_dir, "cores_with_gradient_filled.csv"))
# Select direct measurements only
all = df.trimmed[which(df.trimmed$MethodCM=="direct"),]
all$Depth = as.numeric(all$Depth) # in meters
all$cellsPer = as.numeric(all$cellsPer) # in cell cm-3


# load indices
myIndices = as.matrix(read.csv(file.path(input_dir, "1000_indices_for_bootstrap.csv"),header=FALSE))
myIndices=myIndices
bootstraps = nrow(myIndices)
# using the maximum depth to the 122 °C isotherm, as calculated by the XGBoost model developed in this study
depthsToIterate = gridCells$maxdepth * 1000 

AntarcFID = 3791:4163 # These are FID_1 on TC's documents already corrected in CM
GreenlandFID = c(3774:3776,3759:3762,3737:3742,3711:3715,3681:3685,3639:3643,3584:3588,3522:3525,3452:3454,3378:3380)


temperature.byDepthAll <- matrix(NA_real_, nrow = bootstraps, ncol = nbins)
lm.byDepthAll          <- matrix(NA_real_, nrow = bootstraps, ncol = nbins)

temperature.error = vector(length=bootstraps)
lm.error = vector(length=bootstraps)

temperature.biomass = vector(length=bootstraps)
lm.biomass = vector(length=bootstraps)

temperature.byGridResult = data.frame(matrix(NA, nrow =nrow(gridCells), ncol = bootstraps))
lm.byGridResult = data.frame(matrix(NA, nrow =nrow(gridCells), ncol = bootstraps))

temperature.parameters = data.frame(matrix(NA,nrow=bootstraps,ncol=2))
lm.parameters = data.frame(matrix(NA,nrow=bootstraps,ncol=2))

temperature.r2 = vector(length = bootstraps)
lm.r2 = vector(length = bootstraps)


ptm <- proc.time()
for (n in 1:nrow(myIndices)) {
  if(n%%5==0){ 
    print(n)
  }

  train_ind <- myIndices[n,]
  
 ## Substitute the temperature calculation formula with gradients inferred by the XGBoost model developed in this study.
  all$temperature = all$mast + all$Depth*all$gradient/1000
  
  
  ## Training set for Temperature
  train.temp = all[train_ind,]
  test.temp = all[-train_ind,]
  
  ## Training set for Depth
  pf.train = all[train_ind,]
  pf.test = all[-train_ind,]
  
  ## Temperature Fit
  train.temp$cellsPer = log10(train.temp$cellsPer) 
  test.temp = test.temp[which(complete.cases(test.temp$T..oC.)),]
  test.temp$cellsPer = log10(test.temp$cellsPer) 
  # test.temp$T..oC. = test.temp$temperature
  
  powerFit= lm(cellsPer~T..oC.,train.temp)
  
  powerResult = predict(powerFit,test.temp)
  
  ss_res_temp = sum((test.temp$cellsPer - powerResult)^2)
  ss_tot_temp = sum((test.temp$cellsPer - mean(test.temp$cellsPer))^2)
  temperature.r2[n] = 1 - ss_res_temp / ss_tot_temp
  
  a =  powerFit$coefficients[1]
  slope = powerFit$coefficients[2]
  
  temperature.parameters[n,1] =a
  temperature.parameters[n,2] =slope
  
  biomass = 0
  temp = matrix(NA, nrow = nrow(gridCells), ncol = 1)
  byDepthVec_temp <- rep(0, nbins)

  for (i in 1:nrow(gridCells)) {
    mast     = as.numeric(gridCells$MEAN_Annual_Temp[i])
    gradient = as.numeric(gridCells$gradient[i])
    A_cm2    = gridCells$grid_area_m2[i] * 100 * 100
    zmax     = depthsToIterate[i]
    if (is.na(zmax) || zmax < 1) { temp[i] = 0; next }

    if (gridCells$FID[i] %in% GreenlandFID) {
      # Greenland: T(x) = 0 + x*gradient/1000
      surface_term <- 15000000000
      integralFun <- function(x) { 10^(slope * (0 + (x) * gradient / 1000) + a) }
    } else if (gridCells$FID[i] %in% AntarcFID) {
      surface_term <- 2150000000
      integralFun <- function(x) { 10^(slope * (0 + (x) * gradient / 1000) + a) }
    } else {
      surface_term <- 0
      integralFun <- function(x) { 10^(slope * (mast + (x) * gradient / 1000) + a) }
    }

    grid_sum_all <- 0
    for (k in 1:nbins) {
      z1 <- depth_edges_m[k]
      z2 <- depth_edges_m[k+1]
      if (z1 >= zmax) break
      up <- min(z2, zmax)

      if (up > z1) {
        contrib <- A_cm2 * integrate(integralFun, z1, up)$value * 100
        byDepthVec_temp[k] <- byDepthVec_temp[k] + contrib
        grid_sum_all       <- grid_sum_all + contrib
      }

      if (k == 1 && surface_term > 0) {
        add0 <- A_cm2 * surface_term
        byDepthVec_temp[k] <- byDepthVec_temp[k] + add0
        grid_sum_all       <- grid_sum_all + add0
      }
    }

    temp[i]  = grid_sum_all
    biomass  = biomass + grid_sum_all
  }

  temperature.byGridResult[, n] <- temp
  temperature.biomass[n]        <- biomass
  temperature.byDepthAll[n, ]   <- byDepthVec_temp
  temperature.error[n] = mean((powerResult-test.temp$cellsPer)^2)
  #print("temperature error")
  #print(mean((powerResult-test.temp$cellsPer)^2))
  print("temperature biomass")
  print(biomass)
  
  ## Depth PowerFit 
  pf.train$Depth = log10(as.numeric(pf.train$Depth))
  pf.train$cellsPer = log10(as.numeric(pf.train$cellsPer))
  pf.test$Depth = log10(as.numeric(pf.test$Depth))
  pf.test$cellsPer = log10(as.numeric(pf.test$cellsPer))
  powerFit = lm(cellsPer~Depth,data=pf.train)
  #  pf.x = data.frame(test$Depth)
  powerResult = predict(powerFit,pf.test)

  ss_res_lm = sum((pf.test$cellsPer - powerResult)^2)
  ss_tot_lm = sum((pf.test$cellsPer - mean(pf.test$cellsPer))^2)
  lm.r2[n] = 1 - ss_res_lm / ss_tot_lm
  a =  powerFit$coefficients[1]
  b = powerFit$coefficients[2]
  
  
  lm.parameters[n,1] =a
  lm.parameters[n,2] =b
  #### Integration step 
  biomass = 0
  dep = matrix(NA, nrow = nrow(gridCells), ncol = 1)
  byDepthVec_lm <- rep(0, nbins)

  for (i in 1:nrow(gridCells)) {
    A_cm2 = gridCells$grid_area_m2[i] * 100 * 100
    zmax  = depthsToIterate[i]
    if (is.na(zmax) || zmax < 1) { dep[i] = 0; next }

    if (gridCells$FID[i] %in% GreenlandFID) {
      surface_term <- 15000000000
      integralFun  <- function(x) { (10^a) * x^b }
    } else if (gridCells$FID[i] %in% AntarcFID) {
      surface_term <- 2150000000
      integralFun  <- function(x) { (10^6) * x^b }
    } else {
      surface_term <- 0
      integralFun  <- function(x) { (10^a) * x^b }
    }

    grid_sum_all <- 0
    for (k in 1:nbins) {
      z1 <- depth_edges_m[k]
      z2 <- depth_edges_m[k+1]
      if (z1 >= zmax) break
      up <- min(z2, zmax)

      if (up > z1) {
        contrib <- A_cm2 * integrate(integralFun, z1, up)$value * 100
        byDepthVec_lm[k] <- byDepthVec_lm[k] + contrib
        grid_sum_all     <- grid_sum_all + contrib
      }

      if (k == 1 && surface_term > 0) {
        add0 <- A_cm2 * surface_term
        byDepthVec_lm[k] <- byDepthVec_lm[k] + add0
        grid_sum_all     <- grid_sum_all + add0
      }
    }

    dep[i]  = grid_sum_all
    biomass = biomass + grid_sum_all
  }

  lm.byGridResult[, n] <- dep
  lm.biomass[n]        <- biomass
  lm.byDepthAll[n, ]   <- byDepthVec_lm
  lm.error[n] = mean((powerResult-pf.test$cellsPer)^2)
  print("depth biomass")
  print(biomass)
}
proc.time() - ptm


# Depth-based model outputs
write.table(lm.biomass,file = "lm_with_merged_depth.biomass_md.csv",sep=",",row.names=FALSE,col.names=TRUE)
write.table(lm.error,file = "lm_with_merged_depthtemp122_lm.error_md.csv",sep=",",row.names=FALSE,col.names=TRUE)
write.table(lm.byGridResult, file = 'lm_with_merged_depthtemp122_lm_GridResult_md.csv',sep=',',row.names=FALSE,col.names=TRUE)
write.table(lm.parameters, file = 'lm_with_merged_depthtemp122_lm_parameters_md.csv',sep=',',row.names=FALSE,col.names=TRUE)

write.table(temperature.biomass, file = "temperature_model_total_biomass.csv", sep = ",", row.names = FALSE, col.names = FALSE)
write.table(temperature.error, file = "temperature_model_error.csv", sep = ",", row.names = FALSE, col.names = FALSE)
write.table(temperature.byGridResult, file = "temperature_model_grid_result.csv", sep = ",", row.names = FALSE, col.names = FALSE)
write.table(temperature.parameters, file = "temperature_model_parameters.csv", sep = ",", row.names = FALSE, col.names = FALSE)

temp_by_depth_matrix <- data.frame(
  depth_top_km = depth_edges_km[-length(depth_edges_km)],
  depth_bot_km = depth_edges_km[-1]
)
temp_by_depth_matrix <- cbind(
  temp_by_depth_matrix,
  as.data.frame(t(temperature.byDepthAll))  # iter_1..iter_B
)
colnames(temp_by_depth_matrix)[-(1:2)] <- paste0("iter_", seq_len(bootstraps))
write.csv(temp_by_depth_matrix, file = "temperature_model_biomass_by_depth_matrix.csv", row.names = FALSE)

temp_mean <- apply(temperature.byDepthAll, 2, mean,     na.rm = TRUE)
temp_med  <- apply(temperature.byDepthAll, 2, median,   na.rm = TRUE)
temp_lo  <- apply(temperature.byDepthAll, 2, quantile, probs = 0.025, na.rm = TRUE)
temp_hi  <- apply(temperature.byDepthAll, 2, quantile, probs = 0.975, na.rm = TRUE)
temp_by_depth_summary <- data.frame(
  depth_top_km = depth_edges_km[-length(depth_edges_km)],
  depth_bot_km = depth_edges_km[-1],
  total_mean   = temp_mean,
  total_median = temp_med,
  total_lo95   = temp_lo,
  total_hi95   = temp_hi
)
write.csv(temp_by_depth_summary, file = "temperature_model_biomass_by_depth_summary.csv", row.names = FALSE)

lm_by_depth_matrix <- data.frame(
  depth_top_km = depth_edges_km[-length(depth_edges_km)],
  depth_bot_km = depth_edges_km[-1]
)
lm_by_depth_matrix <- cbind(
  lm_by_depth_matrix,
  as.data.frame(t(lm.byDepthAll))
)
colnames(lm_by_depth_matrix)[-(1:2)] <- paste0("iter_", seq_len(bootstraps))
write.csv(lm_by_depth_matrix, file = "lm_biomass_by_depth_matrix.csv", row.names = FALSE)

lm_mean <- apply(lm.byDepthAll, 2, mean,     na.rm = TRUE)
lm_med  <- apply(lm.byDepthAll, 2, median,   na.rm = TRUE)
lm_lo  <- apply(lm.byDepthAll, 2, quantile, probs = 0.025, na.rm = TRUE)
lm_hi  <- apply(lm.byDepthAll, 2, quantile, probs = 0.975, na.rm = TRUE)
lm_by_depth_summary <- data.frame(
  depth_top_km = depth_edges_km[-length(depth_edges_km)],
  depth_bot_km = depth_edges_km[-1],
  total_mean   = lm_mean,
  total_median = lm_med,
  total_lo95   = lm_lo,
  total_hi95   = lm_hi
)
write.csv(lm_by_depth_summary, file = "lm_biomass_by_depth_summary.csv", row.names = FALSE)
