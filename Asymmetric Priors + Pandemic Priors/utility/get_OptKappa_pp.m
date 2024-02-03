% This function obtains the optimal shrinkage hyperparameter values and the
% associated marginal likelihood value under the asymmetric conjugate prior
%
% See:
% Chan, J.C.C. (2021). Asymmetric conjugate priors for large Bayesian VARs,
% Quantitative Economics, forthcoming.
%
% Input: idx_ns - index for nonstationary variables

function [ml_opt,kappa_opt] = get_OptKappa_pp(Y0,Y,Z,p,k0,type,idx_ns,covid_periods)
if nargin == 6
    idx_ns = [];
end
kappa3 = 1; kappa4 = 100;
n = size(Y,2);
sig2 = get_resid_var(Y0,Y);
if strcmp(type,'stru')
    f = @(k) -ml_VAR_ACP_pp(p,Y,Z,prior_ACP_stru_pp(n,p,[exp(k(1)),exp(k(2)),kappa3,kappa4,exp(k(3))],sig2,idx_ns,covid_periods),covid_periods);
elseif strcmp(type,'redu')
    f = @(k) -ml_VAR_ACP_pp(p,Y,Z,prior_ACP_redu_pp(n,p,[exp(k(1)),exp(k(2)),kappa3,kappa4,exp(k(3))],sig2,idx_ns,covid_periods),covid_periods);
end
[k_opt,nml] = fminsearch(f,log(k0));
ml_opt = -nml;
kappa_opt = [exp(k_opt(1)),exp(k_opt(2)),kappa3,kappa4,exp(k_opt(3))];
end