##################################################################
#
# Replication files of the paper
# "Pandemic Priors", Cascaldi-Garcia, D.
#
# Use of code for research purposes is permitted as long as proper
# reference to source is given
#
# Danilo Cascaldi-Garcia
#
# December/2023
#
##################################################################

library(Matrix)
library(LaplacesDemon)
library(tictoc)
library(ggplot2)
library(reshape2)
library(dplyr)

tic("Running time:")
source("~/R/Pandemic Priors/functions_Pandemic_Priors.R") # Include the auxiliary functions

# Define specification of the VAR model
constant <- 1                   # 0 --> no intercepts, 1 --> with intercepts
nAR <- 12                       # Lags of the VAR
nimp <- 36                      # horizon of IRFs to be shown
rps <- 1000                     # Number of coefficient draws
covid_periods <- 6              # Number of COVID-19 periods for time dummies, starting from March/2020; set to zero for no COVID-19 dummies
diff_or_lv <- 1                 # 0 --> in differences, 1 --> level
test_stab <- 1                  # 0 --> all posterior draws, 1 --> only stationary draws
nshocks <- 1                    # Number of identified shocks (Cholesky ordering)
log_vector <- c(0, 1, 0, 1, 1, 1, 1, 0) # Variables in log
bands <- c(50, 16, 84)          # Coverage bands
savefigures <- 1                # 1 --> Save figures as .PDF

# Specify parameters of the Pandemic Priors
λ <- 0.2                          # overall prior tightness
ϵ <- 0.001                        # prior for the constant
ϕ <- 999                          # prior for the pandemic; 999 = optimal, 0.001 = uninformative

####################### Monthly Data #######################

setwd("C:/R/Pandemic Priors") # set your folder path
b1 <- readxl::read_excel("Data.xlsx", sheet = "Sheet1")
Ynames <- c("EBP", "S&P 500", "Shadow Rate", "PCE", "PCE Price Index", "Employment", "Ind. Production", "Unemp. Rate")
time_vec <- seq(as.Date("1975-01-01"), as.Date("2022-12-01"), by = "month")

#############################################################

Yraw <- as.matrix(b1[,2:ncol(b1)])
for (ee in seq_along(log_vector)) {
  if (log_vector[ee]==1) { Yraw[,ee] <- log(Yraw[,ee])*100 }
}
if (diff_or_lv ==0) {
  Yraw_temp <- Yraw[2:nrow(Yraw),]
  for (ee in seq_along(log_vector)) {
    if (log_vector[ee]==1) { Yraw_temp[,ee] <- diff(Yraw[,ee], differences=1) }
  }
  Yraw <- Yraw_temp
}
Traw <- nrow(Yraw); n <- ncol(Yraw)
x <- mlag2(Yraw,nAR)
if (constant==1) {  X1 <- cbind(x[(nAR+1):Traw,], rep(1, Traw-nAR)) } else { X1 <- x[(nAR+1):Traw,] }

## COVID-19 time dummies
X1 <- cbind(X1, matrix(0, nrow=nrow(X1), ncol=covid_periods))
if (diff_or_lv ==1) {
  covid_ind <- which(time_vec == as.Date("2020-03-01"))-nAR
} else {
  covid_ind <- which(time_vec == as.Date("2020-03-01"))-nAR-1
}
X1[(covid_ind):(covid_ind+covid_periods-1), (ncol(X1)-covid_periods+1):ncol(X1)] <- diag(covid_periods)

## Adjusting matrices
Y1 <- Yraw[(nAR+1):Traw,]; T <- Traw - nAR; Y <- Y1; X <- X1; K <- ncol(X)

# OLS coefficients
A_OLS <- solve(t(X)%*%X)%*%t(X)%*%Y
SSE <- t(Y - X%*%A_OLS)%*%(Y - X%*%A_OLS); SIGMA_OLS <- SSE/(T-K);
RESID_OLS <- (Y - X%*%A_OLS); VCV_OLS <- cov(RESID_OLS); A0_OLS <- chol(VCV_OLS); A0_OLS <- t(A0_OLS)

# Priors
λ <- as.numeric(λ)
if (diff_or_lv ==0) { δ <- 0 } else if (diff_or_lv ==1) { δ <- 1 }  # prior mean of the coefficient matrix
τ <- 10*λ;
δ <- as.numeric(δ); ϵ <- as.numeric(ϵ)
ϕ <- as.numeric(ϕ)

if (ϕ == 999) {
  # Define grid of ϕ values
  ϕ_temp <- c(0.001, 0.01, 0.025, 0.050, 0.075, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.75, 1, 2, 5)
  ϕTest <- rep(0, length(ϕ_temp))
  
  # Iterate over grid of ϕ values
  for (jj in seq_along(ϕ_temp)) {
    # Calculate density for current ϕ value
    Density <- OptimalPhi(X, Y, Yraw, nAR, constant, δ, λ, τ, ϵ, ϕ_temp[jj], covid_periods)
    # Store density in ϕTest vector
    ϕTest[jj] <- Density
    print(paste("Density(",ϕ_temp[jj],") = ", Density))
  }
  
  # Find maximum density and corresponding ϕ
  maxDensity <- max(ϕTest)
  GridPosition <- which.max(ϕTest)
  
  # Set optimal ϕ value
  ϕ_use <- ϕ_temp[GridPosition]
  print(paste("Optimal ϕ for the model is : ", ϕ_use))
} else {
  # Use predefined ϕ value
  ϕ_use <- ϕ
}

# Pandemic Priors and posterior mean
result <- pandemicpriors(X,Y,Yraw,nAR,constant,δ,λ,τ,ϵ,ϕ_use,covid_periods)

# Extract the variables
Xst <- result$Xst
Yst <- result$Yst
xd <- result$xd
yd <- result$yd

XXst = t(xd) %*% xd + t(X) %*% X
invXXst = solve(XXst)
XYst = t(xd) %*% yd + t(X) %*% Y
A_post = invXXst %*% XYst
RESID <- Yst - Xst %*% A_post
SSE_post <- t(RESID) %*% RESID
inv_SSE_post <- solve(SSE_post)

# Posterior draws
v0 <- n + 2
v1 <- dim(Xst)[1] + 2 - dim(Xst)[2]
xx <- t(Xst) %*% Xst
ixx <- solve(xx)
chol_ixx <- chol(ixx)
A_companion_T <- array(0, dim=c(rps, n*nAR, n*nAR))
A0hat_T <- array(0, dim = c(rps, n, n))
pb = txtProgressBar(min = 1, max = rps, initial = 1)
print("Drawing coefficients:")
for (iii in 1:rps) {
  setTxtProgressBar(pb,iii)
  if (test_stab == 0) {
    res <- draw_coef_pandemic_priors(SSE_post, v1, chol_ixx, A_post, n, nAR, covid_periods)
  } else if (test_stab == 1) {
    res <- draw_coef_pandemic_priors_stab(SSE_post, v1, chol_ixx, A_post, n, nAR, covid_periods)
  }
  A_companion_T[iii, , ] <- res$A_companion_dr
  A0hat_T[iii, , ] <- res$A0hat
}
close(pb)

# Calculate IRFs
outs <- array(0, dim = c(rps, nshocks, nimp, n))
pb = txtProgressBar(min = 1, max = rps, initial = 1)
print("Calculating IRFs:")
for (iii in 1:rps) {
  setTxtProgressBar(pb,iii)
  outs[iii, , , ] <- do_irfs(A_companion_T[iii, , ], A0hat_T[iii, , ], n, nshocks, nimp)
}
close(pb)

# Plot settings
quantiles <- def_quantiles(outs, bands)
selec <- 1:nshocks
xaxis <- 1:nimp
line_color <- "black"
fill_color <- "grey"

# Create a data frame
quantiles_df <- melt(drop(quantiles), varnames = c("Line", "Horizon", "Variable"))
quantiles_df <- quantiles_df %>%
  rename(Yname = Variable)
quantiles_df$Yname <- rep(Ynames, each = nrow(quantiles_df) / length(Ynames))

# Separate data frames for median line and ribbon
median_df <- subset(quantiles_df, Line == 1)
ribbon_df <- subset(quantiles_df, Line %in% c(2, 3))

# Calculate upper and lower bounds for ribbon
ribbon_bounds <- ribbon_df %>%
  group_by(Yname, Horizon) %>%
  summarize(ymin = min(value), ymax = max(value), .groups = 'drop')

# Plot
my_plot <- ggplot() +
  geom_ribbon(data = ribbon_bounds,
              aes(x = Horizon, ymin = ymin, ymax = ymax),
              alpha = 0.5, fill = fill_color, color = fill_color) +
  geom_line(data = median_df,
            aes(x = Horizon, y = value),
            linewidth = 1, color = line_color) +
  facet_wrap(~Yname, scales = 'free_y', ncol = 3, strip.position = "bottom") +
  geom_hline(yintercept = 0, linetype = "dotted", color = "black") +  # Add dotted line
  theme_minimal()

# Print and save it
print(my_plot)
if (savefigures == 1) {
  ggsave("output_plot.pdf", my_plot, width = 10, height = 8, units = "in")
}

toc()