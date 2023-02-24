function [Xst, Yst, xd, yd] = pandemicpriors(X,Y,Yraw,nAR,constant,delta,lambda,tau,epsilon,phi,covid_periods)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Replication files of the paper
% "Pandemic Priors", Cascaldi-Garcia, D.
%
% Use of code for research purposes is permitted as long as proper
% reference to source is given
%
% Danilo Cascaldi-Garcia
%
% November/2022
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

T = size(X,1); Traw = T+nAR; n=size(Y,2);
mu=mean(Y); % Banbura et al (2010) average over the entire sample

%% Construct a univariate autoregressive model of order nAR for each variable
Coef = NaN(nAR+1,n);
ResidAR = NaN(length(Yraw(nAR+1:Traw,:)),n);
for i=1:n
    YlagAR = NaN(length(Yraw),nAR);
    for ii=1:nAR; YlagAR(nAR+1:Traw,ii)=Yraw(nAR+1-ii:Traw-ii,i); end
    X_AR = [YlagAR(nAR+1:Traw,:) ones(Traw-nAR,1)];
    Y_AR = Yraw(nAR+1:Traw,i);
    A_AR = (X_AR'*X_AR)\(X_AR'*Y_AR);
    Coef(:,i) = A_AR; ResidAR(:,i) = Y_AR - X_AR*Coef(:,i);
end

SSE2 = (ResidAR)'*(ResidAR); sizeAR = size(ResidAR,1);
SIGMA_AR = SSE2./sizeAR; sigma = sqrt(SIGMA_AR);

%% Construct dummies for the pandemic priors
yd1 = []; yd2 = []; xd1 = []; xd2 = [];

if lambda>0
    aa = (diag(sigma.*delta)./lambda);
    s1 = size(aa,1);
    bb = eye(s1);
    jp=diag(1:nAR);
    for i=1:s1; bb(i,i) = aa(i); end
    cc = diag(sigma);
    dd = eye(s1);
    for i=1:s1; dd(i,i) = cc(i); end
    ee = (diag(sigma)./lambda);
    ff = eye(s1);
    for i=1:s1; ff(i,i) = ee(i); end
    if constant == 1
        yd1=[bb; zeros(n*(nAR-1),n); dd; zeros(covid_periods+1,n)];
        xd1=[kron(jp,ff) zeros((n*nAR),covid_periods+1); zeros(n,(n*nAR)+covid_periods+1);
            zeros(1,n*nAR) epsilon zeros(1,covid_periods);
            zeros(covid_periods,n*nAR+1) diag(ones(1,covid_periods)*phi)];
    else
        yd1=[bb; zeros(n*(nAR-1),n); dd; zeros(covid_periods,n)];
        xd1=[kron(jp,ff) zeros((n*nAR),covid_periods); zeros(n,(n*nAR)+covid_periods);
            zeros(covid_periods,n*nAR) diag(ones(1,covid_periods)*phi)];
    end

%% Construct sum of coefficient dummies
    if tau>0
        Vec1 = ones(1,nAR);
        aa2 = diag(delta.*mu)./tau;
        s1 = size(aa2,1);
        bb2 = eye(s1);
        for i=1:s1; bb2(i,i) = aa2(i,i); end
        yd2=bb2;
        if constant == 1
            xd2=[kron(Vec1,yd2) zeros(n,covid_periods+1)];
        else
            xd2=[kron(Vec1,yd2) zeros(n,covid_periods)];
        end
    else
        yd2 = [];
        xd2 = [];
    end
end

yd=[yd1;yd2]; xd=[xd1;xd2]; Yst = [Y; yd]; Xst = [X; xd];
