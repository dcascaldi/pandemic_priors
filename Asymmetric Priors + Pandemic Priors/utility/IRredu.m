% This function computes the imuplse responses
%
% See:
% Chan, J.C.C. (2021). Asymmetric conjugate priors for large Bayesian VARs,
% Quantitative Economics, forthcoming.
%
% Inputs:
% A: reduced-form coef., each column contains coef. for each equation
% L: L*L' = reduced-form covariance matrix

function response = IRredu(A,L,nstep,nshock)
[np, n] = size(A);
p = np/n;
response = zeros(n,nshock,nstep);
Acomp = [A'; sparse(1:n*(p-1),1:n*(p-1),ones(n*(p-1),1),n*(p-1),np)];
response(:,:,1) = L(:,1:nshock);
Apower = Acomp;
for it = 2:nstep
    Apower = Apower*Acomp;    
    response(:,:,it) = Apower(1:n,1:n)*L(:,1:nshock);
end    
end