# Associate functions for the Pandemic Priors
# Created by Danilo Cascaldi-Garcia
# Federal Reserve Board

mlag2 <- function(X, p) {
  # Created by Danilo Cascaldi-Garcia
  Traw <- nrow(X)
  N <- ncol(X)
  Xlag <- matrix(0, Traw, N*p)
  for (ii in 1:p) {
    Xlag[(p+1):Traw, ((N*(ii-1)+1):(N*ii))] <- X[(p+1-ii):(Traw-ii), 1:N]
  }
  return(Xlag)
}

mlag1 <- function(X, p) {
  # Created by Danilo Cascaldi-Garcia
  Traw <- length(X)
  Xlag <- matrix(0, Traw, p)
  for (ii in 1:p) {
    Xlag[(p+1):Traw, ((ii-1)+1):ii] <- X[(p+1-ii):(Traw-ii)]
  }
  return(Xlag)
}

pandemicpriors <- function(X, Y, Yraw, nAR, constant, delta, lambda, tau, ϵ, phi, covid_periods) {
  # Created by Danilo Cascaldi-Garcia
  ϵ <- as.numeric(ϵ)
  T <- nrow(X)
  Traw <- T + nAR
  p <- nAR
  mu <- apply(Y, 2, mean)
  n <- ncol(Y)
  # Construct a univariate autoregressive model of order p for each n
  Coef <- matrix(0, p+1, n)
  ResidAR <- matrix(0, Traw-nAR, n)
  for (i in 1:n) {
    YlagAR <- mlag1(Yraw[,i], p)
    X_AR <- cbind(YlagAR[(nAR+1):Traw,], rep(1, Traw-nAR))
    Y_AR <- Yraw[(nAR+1):Traw,i]
    A_AR <- solve(t(X_AR) %*% X_AR) %*% t(X_AR) %*% Y_AR
    Coef[,i] <- A_AR
    ResidAR[,i] <- Y_AR - X_AR %*% Coef[,i]
  }
  SSE2 <- t(ResidAR) %*% ResidAR
  sizeAR <- nrow(ResidAR)
  sigma_AR <- SSE2/sizeAR
  sigma <- matrix(0, nrow(sigma_AR), ncol(sigma_AR))
  for (ll in 1:nrow(sigma_AR)) {
    sigma[ll,ll] <- sqrt(sigma_AR[ll,ll])
  }
  if (lambda > 0) {
    bb <- diag(diag(sigma*delta)/lambda)
    dd <- diag(diag(sigma))
    ff <- diag(diag(sigma)/lambda)  
    jp <- diag(1:p)
    if (constant == 1) {
      yd1 <- rbind(bb, matrix(0, n*(p-1), n), dd, matrix(0, covid_periods+1, n))
      xd1 <- rbind(
        cbind(kronecker(jp, ff), diag(0, n * p, covid_periods + 1)),
        diag(0, n, n * p + covid_periods + 1),
        cbind(diag(0, 1, n * p), c(ϵ), diag(0, 1, covid_periods)),
        cbind(diag(0, covid_periods, n * p+1), diag(rep(1, covid_periods)) * phi)
        )
    } else {
      yd1 <- rbind(bb, matrix(0, n*(p-1), n), dd, matrix(0, covid_periods, n))
      xd1 <- rbind(
        cbind(kronecker(jp, ff), diag(0, n * p, covid_periods)),
        diag(0, n, n * p + covid_periods),
        cbind(diag(0, covid_periods, n * p), diag(rep(1, covid_periods)) * phi)
      )
    }              
    if (tau > 0) {
      bb2 <- diag(delta*mu/tau)
      yd2 <- bb2
      if (constant == 1) {
        xd2 <- cbind(kronecker(matrix(1, 1, p), yd2), matrix(0, n, covid_periods+1))
      } else {
        xd2 <- cbind(kronecker(matrix(1, 1, p), yd2), matrix(0, n, covid_periods))
      }
    } else {
      yd2 <- matrix(0, 0, 0)
      xd2 <- matrix(0, 0, 0)               
    }
  }
  yd <- rbind(yd1, yd2)
  xd <- rbind(xd1, xd2)
  Yst <- rbind(Y, yd)
  Xst <- rbind(X, xd)
  return(list("Xst" = Xst, "Yst" = Yst, "xd" = xd, "yd" = yd))
}

draw_coef_pandemic_priors_stab <- function(SSE_post, v1, chol_ixx, A_post, n, nAR, covid_periods) {
  # Created by Danilo Cascaldi-Garcia
  control2 <- 0
  while (control2 == 0) {
    sigma_rep <- rinvwishart(v1, SSE_post) # draw σ
    chol_sigma_rep <- chol(sigma_rep)
    nbeta_dr <- A_post + t(chol_ixx) %*% matrix(rnorm(n*nAR+covid_periods+1), (n*nAR+covid_periods+1), n) %*% chol_sigma_rep
    A0hat <- t(chol_sigma_rep)
    A_companion_dr <- matrix(0, n*nAR, n*nAR)
    A_companion_dr[1:n,] <- t(nbeta_dr[1:c(n*nAR),])
    A_companion_dr[c(n+1):c(n*nAR),(1:c(n*nAR-n))] <- diag(c(n*nAR-n))
    eigAR_a <- max(abs(eigen(A_companion_dr)$values))
    if (eigAR_a < 1.01) {
      control2 <- 1
      return(list("A_companion_dr" = A_companion_dr, "A0hat" = A0hat))
    } else {
      next
    }
  } 
}

draw_coef_pandemic_priors <- function(SSE_post, v1, chol_ixx, A_post, n, nAR, covid_periods) {
  # Created by Danilo Cascaldi-Garcia
  sigma_rep <- rinvwishart(v1, SSE_post) # draw σ
  chol_sigma_rep <- chol(sigma_rep)
  nbeta_dr <- A_post + t(chol_ixx) %*% matrix(rnorm(n*nAR+covid_periods+1), (n*nAR+covid_periods+1), n) %*% chol_sigma_rep
  A0hat <- t(chol_sigma_rep)
  A_companion_dr <- matrix(0, n*nAR, n*nAR)
  A_companion_dr[1:n,] <- t(nbeta_dr[1:c(n*nAR),])
  A_companion_dr[c(n+1):c(n*nAR),(1:c(n*nAR-n))] <- diag(c(n*nAR-n))
  return(list("A_companion_dr" = A_companion_dr, "A0hat" = A0hat))
}

do_irfs <- function(A_companion_dr, IMP_dr, n, nshocks, nimp) {
  U1_dr <- rbind(IMP_dr, matrix(0, c(n * nAR - n), n))
  Zk1_dr <- array(0, dim = c(nshocks, nimp, n * nAR))
  impulse1_dr <- array(0, dim = c(nshocks, nimp, n))
  for (r in 1:nshocks) {
    for (k in 1:nimp) {
      if (k ==1) {A_temp = diag(n * nAR)}
      else {A_temp = A_temp %*% A_companion_dr} # This is faster than power matrix
      Zk1_dr[r, k, ] <- t(A_temp %*% U1_dr[, r])
    }
    impulse1_dr[r, , ] <- Zk1_dr[r, , 1:n]
    if (impulse1_dr[r, 1, r] < 0) {
      impulse1_dr[r, , ] <- impulse1_dr[r, , ] * (-1)
    }
  }
  return(impulse1_dr)
}

def_quantiles <- function(out, prob) {
  # Created by Danilo Cascaldi-Garcia
  shocks <- dim(out)[2]; hor <- dim(out)[3]; var_ef <- dim(out)[4]
  nprob <- length(prob)
  quantiles_out <- array(0, dim=c(nprob, shocks, hor, var_ef))  
  for (g in 1:nprob) {
    for (i in 1:shocks) {
      for (h in 1:hor) {
        for (j in 1:var_ef) {
          quantiles_out[g,i,h,j] <- quantile(out[,i,h,j], prob[g]/100)
        }
      }
    }
  }
  return(quantiles_out)
}

OptimalPhi <- function(X, Y, Yraw, nAR, constant, δ, λ, τ, ϵ, ϕ, covid_periods) {
  # Convert arrays to matrices
  X <- as.matrix(X)
  Y <- as.matrix(Y)
  Yraw <- as.matrix(Yraw)
  
  # Pandemic Priors
  Res <- pandemicpriors(X, Y, Yraw, nAR, constant, δ, λ, τ, ϵ, ϕ, covid_periods)
  xd <- as.matrix(Res$xd)
  yd <- as.matrix(Res$yd)
  T <- nrow(X)
  n <- ncol(Y)
  
  # Dummies
  xx0 <- t(xd) %*% xd
  ixx0 <- solve(xx0)
  b0 <- ixx0 %*% t(xd) %*% yd
  v0 <- n + 2
  e0 <- yd - xd %*% b0
  σ0 <- t(e0) %*% e0
  v1 <- v0 + T
  
  # Density
  aPP <- diag(T) + X %*% ixx0 %*% t(X)
  PP <- solve(aPP)
  QQ <- σ0

  # Log n-variate Gamma function
  # r1
  x_log <- (matrix(1, n, 1) * (v0 / 2)) + ((matrix(1, n, 1) - (1:1:n)) / 2)
  sum_log <- sum(lgamma(x_log))
  r1 <- (n * (n - 1)) / 4 * log(pi) + sum_log
  
  # r2
  x_log <- (matrix(1, n, 1) * (v1 / 2)) + ((matrix(1, n, 1) - (1:1:n)) / 2)
  sum_log <- sum(lgamma(x_log))
  r2 <- (n * (n - 1)) / 4 * log(pi) + sum_log
  
  # p(Y)
  py <- -((T * n) / 2) * log(pi) + (1 / 2) * n * log(det(PP)) + (v0 / 2) * log(det(QQ)) + (r2 - r1) - (v1 / 2) * log(det(QQ + t(Y - X %*% b0) %*% PP %*%  (Y - X %*% b0)))

return(py)
}


