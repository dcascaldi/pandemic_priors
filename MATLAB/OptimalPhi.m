function py=OptimalPhi(X,Y,Yraw,nAR,constant,delta,lambda,tau,epsilon,phi,covid_temp)
[~, ~, xd, yd] = pandemicpriors(X,Y,Yraw,nAR,constant,delta,lambda,tau,epsilon,phi,covid_temp);
    
%======================= PREDICTIVE INFERENCE =============================
%==========================================================================
T = size(X,1);
n = size(Y,2);

% Dummies
xx0 = xd'*xd;
ixx0 = xx0\eye(size(xx0,2));  %inv(xx0'xx0)
b0 = ixx0*xd'*yd;

v0 = n+2;
e0 = yd-xd*b0;
sigma0 = e0'*e0;

% Posterior
v1 = v0+T;

% Density
aPP = (eye(T) + X*ixx0*X');
PP = aPP\eye(size(aPP,2)); %inv(aPP)
QQ = sigma0;

%% Log n-variate Gamma function
% r1
x_log = ones(n,1)*(v0/2) + (1-(1:n)')/2;
r1 = n*(n-1)/4*log(pi)+sum(gammaln(x_log),1);

% r2
x_log = ones(n,1)*(v1/2) + (1-(1:n)')/2;
r2 = n*(n-1)/4*log(pi)+sum(gammaln(x_log),1);

%% p(Y)
 py = -(T*n/2)*log(pi) + (1/2)*n*log(det(PP)) + (v0/2)*log(det(QQ)) +...
     (r2-r1) - (v1/2)*log(det(QQ + (Y-X*b0)'*PP*(Y-X*b0)));