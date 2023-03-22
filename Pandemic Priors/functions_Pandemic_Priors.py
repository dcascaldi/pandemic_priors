# Associate functions for the Pandemic Priors
# Created by Danilo Cascaldi-Garcia
# Federal Reserve Board

import numpy as np
from scipy.stats import invwishart
from scipy.linalg import cholesky, eig
from scipy.special import loggamma

# def mlag2(X, p):
#     # Created by Danilo Cascaldi-Garcia
#     Traw, N = X.shape
#     Xlag = np.zeros((Traw, N*p))
#     for ii in range(1, p+1):
#         Xlag[p:Traw, N*(ii-1):N*ii] = X[p-ii:Traw-ii, :]
#     return Xlag

# def mlag1(X, p):
#     # Created by Danilo Cascaldi-Garcia
#     Traw = X.shape[0]
#     Xlag = np.zeros((Traw, p))
#     for ii in range(1, p+1):
#         Xlag[p:Traw, ii-1:ii] = X[p-ii:Traw-ii]
#     return Xlag

def pandemicpriors(X, Y, Yraw, nAR, constant, delta, lam, tau, eps, phi, covid_periods):
    # Created by Danilo Cascaldi-Garcia
    T = X.shape[0]
    Traw = T + nAR
    p = nAR
    mu = np.mean(Y, axis=0)
    n = Y.shape[1]
    
    # Construct a univariate autoregressive model of order p for each n
    Coef = np.empty((p+1, n))
    ResidAR = np.empty((Traw-nAR, n))
    for i in range(n):
        YlagAR = np.empty((Traw, 1 * nAR)) * np.nan
        for ii in range(nAR):
            YlagAR[nAR:Traw, ii] = Yraw[nAR-1-ii:Traw-1-ii,i]
        X_AR = np.concatenate((YlagAR[nAR:Traw,:], np.ones((Traw - nAR, 1))), axis=1)
        Y_AR = Yraw[nAR:Traw, i]
        A_AR = np.linalg.inv(X_AR.T @ X_AR) @ (X_AR.T @ Y_AR)
        Coef[:, i] = A_AR
        ResidAR[:, i] = Y_AR - X_AR @ Coef[:, i]
    
    SSE2 = np.sum(ResidAR**2, axis=0)
    sizeAR = ResidAR.shape[0]
    sigma_AR = SSE2 / sizeAR
    sigma = np.zeros((sigma_AR.shape[0],sigma_AR.shape[0]))
    for ll in range(sigma_AR.shape[0]):
        sigma[ll, ll] = np.sqrt(sigma_AR[ll])
    # print(SSE2)

    if lam > 0:
        bb = np.diag(np.diag(sigma * delta) / lam)
        dd = np.diag(np.diag(sigma))
        ff = np.diag(np.diag(sigma) / lam)
        jp = np.diag(np.arange(1, p+1))
        if constant == 1:
            yd1 = np.vstack([bb, np.zeros((n*(p-1), n)), dd, np.zeros((covid_periods+1, n))]) 
            xd1 = np.vstack([np.hstack([np.kron(jp, ff), np.zeros((n*p, covid_periods+1))]), 
                             np.zeros((n, n*p + covid_periods+1)),
                             np.hstack([np.zeros((1, n*p)), np.ones((1,1))*eps, np.zeros((1, covid_periods))]),
                             np.hstack([np.zeros((covid_periods, n*p+1)), np.diag(np.ones(covid_periods))*phi])
                             ])
        else:
            yd1 = np.vstack([bb, np.zeros((n*(p-1), n)), dd, np.zeros((covid_periods, n))]) 
            xd1 = np.vstack([np.hstack([np.kron(jp, ff), np.zeros((n*p, covid_periods))]), 
                             np.zeros((n, n*p + covid_periods)),
                             np.hstack([np.zeros((1, n*p)), np.zeros((1, covid_periods))]),
                             np.hstack([np.zeros((covid_periods, n*p+1)), np.diag(np.ones(covid_periods))*phi])
                             ])
        if tau > 0:
            bb2 = np.diag((delta*mu) / tau)
            yd2 = bb2
            # yd2 = bb2.reshape(-1, 1)
            if constant == 1:
                xd2 = np.hstack([np.kron(np.ones((1, p)), yd2), np.zeros((n, covid_periods+1))])
            else:
                xd2 = np.hstack([np.kron(np.ones((1, p)), yd2), np.zeros((n, covid_periods))])
        else:
            yd2 = np.array([])
            xd2 = np.array([])
            
    yd = np.vstack([yd1, yd2])
    xd = np.vstack([xd1, xd2])
    Yst = np.vstack([Y, yd])
    Xst = np.vstack([X, xd])
    return Xst, Yst, xd, yd

def draw_coef_pandemic_priors_stab(SSE_post: np.ndarray, v1: int, chol_ixx: np.ndarray, A_post: np.ndarray, n: int, nAR: int, covid_periods: int):
    # Created by Danilo Cascaldi-Garcia
    control2 = 0
    while control2 == 0:
        sigma_rep = invwishart.rvs(v1, SSE_post, random_state=None)
        chol_rep = cholesky(sigma_rep)
        nbeta_dr = A_post + chol_ixx.T @ np.random.randn(n*nAR+covid_periods+1, n) @ chol_rep
        A0hat = chol_rep.T
        A_companion_dr = np.zeros((n*nAR, n*nAR))
        A_companion_dr[:n,:] = nbeta_dr[:n*nAR,:].T
        A_companion_dr[n:n*nAR,:n*nAR-n] = np.eye(n*nAR-n)
        eigvals, _ = eig(A_companion_dr)
        eigAR_a = np.max(np.abs(eigvals))
        if eigAR_a < 1.01:
            control2 = 1
            return A_companion_dr, A0hat
        else:
            continue

def draw_coef_pandemic_priors(SSE_post: np.ndarray, v1: int, chol_ixx: np.ndarray, A_post: np.ndarray, n: int, nAR: int, covid_periods: int):
    # Created by Danilo Cascaldi-Garcia
    sigma_rep = invwishart.rvs(v1, SSE_post, random_state=None)
    chol_rep = cholesky(sigma_rep)
    nbeta_dr = A_post + chol_ixx.T @ np.random.randn(n*nAR+covid_periods+1, n) @ chol_rep
    A0hat = chol_rep.T
    A_companion_dr = np.zeros((n*nAR, n*nAR))
    A_companion_dr[:n,:] = nbeta_dr[:n*nAR,:].T
    A_companion_dr[n:n*nAR,:n*nAR-n] = np.eye(n*nAR-n)
    return A_companion_dr, A0hat

def do_irfs(A_companion_dr, IMP_dr, n, nshocks, nimp, nAR):
    # Created by Danilo Cascaldi-Garcia
    U1_dr = np.vstack((IMP_dr, np.zeros((n*nAR-n, n))))
    nnn_dr = U1_dr.shape[0]
    Eye_comp_dr = np.eye(nnn_dr) - A_companion_dr
    Zk1_dr = np.zeros((nshocks, nimp, n*nAR))
    impulse1_dr = np.zeros((nshocks, nimp, n))
    for r in range(nshocks):
        for k in range(nimp):
            Zk1_dr[r, k, :] = (np.linalg.matrix_power(A_companion_dr,k) @ U1_dr[:, r]).T
        impulse1_dr[r, :, 0:] = Zk1_dr[r, :, 0:n]
        if impulse1_dr[r, 0, r] < 0:
            impulse1_dr[r, :, :] = np.multiply(impulse1_dr[r, :, :],-1)
    return impulse1_dr

def def_quantiles(out, prob):
    # Created by Danilo Cascaldi-Garcia
    shocks, hor, var_ef = out.shape[1:]
    nprob = prob.shape[0]
    quantiles_out = np.zeros((nprob, shocks, hor, var_ef))
    for g in range(nprob):
        for i in range(shocks):
            for h in range(hor):
                for j in range(var_ef):
                    quantiles_out[g, i, h, j] = np.percentile(out[:, i, h, j], prob[g])
    return quantiles_out

def OptimalPhi(X, Y, Yraw, nAR, constant, delta, lamda, tau, epsilon, phi, covid_periods):
    _, _, xd, yd = pandemicpriors(X, Y, Yraw, nAR, constant, delta, lamda, tau, epsilon, phi, covid_periods)
    T = X.shape[0]
    n = Y.shape[1]
    # Dummies
    xx0 = xd.T @ xd
    ixx0 = np.linalg.inv(xx0)
    b0 = ixx0 @ xd.T @ yd
    v0 = n + 2
    e0 = yd - xd @ b0
    sigma0 = e0.T @ e0   
    # Posterior
    v1 = v0 + T   
    # Density
    aPP = np.eye(T) + X @ ixx0 @ X.T
    PP = np.linalg.inv(aPP)
    QQ = sigma0   
    # Log n-variate Gamma function
    # r1 
    x_log = np.ones((1,n)) * (v0/2) + (np.ones((1,n)) - np.arange(1, n+1))/2
    sum_log = np.sum(loggamma(x_log), axis=1)
    r1 = n * (n - 1) / 4 * np.log(np.pi) + sum_log
    # r2
    x_log = np.ones((1,n)) * (v1/2) + (np.ones((1,n)) - np.arange(1, n+1))/2
    sum_log = np.sum(loggamma(x_log), axis=1)
    r2 = n * (n - 1) / 4 * np.log(np.pi) + sum_log
    # p(Y)
    py = -(T*n/2)*np.log(np.pi) + (1/2)*n*np.log(np.linalg.det(PP)) + (v0/2)*np.log(np.linalg.det(QQ)) + (r2-r1) - (v1/2)*np.log(np.linalg.det(QQ + (Y - X @ b0).T @ PP @ (Y - X @ b0)))
    return py
