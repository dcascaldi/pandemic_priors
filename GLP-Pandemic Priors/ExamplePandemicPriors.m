%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Example of estimating the GLP-Pandemic Priors BVAR
% Giannone, Lenza and Primiceri (2012) + Cascaldi-Garcia (2023)
%
% November/2023
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear
close all
addpath([cd '/Data'])  %on a MAC / Linux
%addpath([cd '\Data']) %on a PC

%% Settings
% Define specification of the BVAR
lags = 12;           % Lags
hmax = 20;          % IRF and forecasts horizon
rps = 10000;        % Number of coefficient draws
covid_periods = 6;  % Number of COVID-19 periods to dummy out, starting from March/2020

%% Load data
data_original = readmatrix("Data2023.xlsx");

% Variable names
Yname = {'EBP','S\&P 500','Shadow Rate','PCE','PCE Price Index','Employment','Ind. Production','Unemp. Rate'};

% Set the sample period
time_vec = datetime(1975,1,1):calmonths(1):datetime(2022,12,1);
data = data_original(1:length(time_vec),2:end);

% Adjust for logs
log_vector = [0 1 0 1 1 1 1 0]; % 1 for variable in logs
for ee=1:size(log_vector,2); if log_vector(ee)==1; data(:,ee) = log(data(:,ee))*100; end; end

%% Pandemic indicator
covid_ind = datefind(datetime(2020,3,1),time_vec,1)-lags; % Find March/2020

%% Estimation
res = bvarGLP_pp(data,lags,covid_ind,covid_periods,'mcmc',1,'MCMCconst',1,'MNpsi',1,'sur',1,'noc',1,'Ndraws',2*rps,'hyperpriors',1,'MCMCfcast',1,'hz',hmax);

%% Posterior distribution of lambda and phi
figure(1)
subplot(2,1,1)
histogram(res.mcmc.lambda)
title('Posterior of the overall shrinkage')
subplot(2,1,2)
h = histogram(res.mcmc.phi);
ax = ancestor(h, 'axes'); ax.XAxis.Exponent = 0;
title('Posterior of phi')

%% Impulse response functions to an EBP shock
nshock = 1; % Position of the variable to shock; 1 is the EBP
nvar = size(data,2);

% IRFs at the posterior mode
beta = res.postmax.betahat(1:nvar*lags+1,:);
sigma = res.postmax.sigmahat;
irf =  bvarIrfs(beta,sigma,nshock,hmax);

% IRFs at each draw
ndraws = size(res.mcmc.beta,3);
Dirf = zeros(hmax,size(data,2),ndraws);
for jg = 1:ndraws
    beta  = res.mcmc.beta(1:nvar*lags+1,:,jg);
    sigma = res.mcmc.sigma(:,:,jg);
    Dirf(:,:,jg) =  bvarIrfs(beta,sigma,nshock,hmax);
end
sIRF = sort(Dirf,3);

figure(2)
%plots the IRFs to an EBP Shock
for jn = 1:nvar
    subplot(2,4,jn)
    plot(0:hmax-1,irf(:,jn),0:hmax-1,squeeze(sIRF(:,jn,round([.16 .5 .84]*ndraws))),'-.r')
    title(Yname{jn})
end

%% Forecasts
figure(3)
%plots the forecasts
sFCST = sort(res.mcmc.Dforecast,3);
hz = res.setpriors.hz(end);
for jn = 1:nvar
    subplot(2,4,jn)
    plot(0:hz-1,res.postmax.forecast(:,jn),0:hz-1,squeeze(sFCST(:,jn,round([.16 .5 .84]*ndraws))),'-.r')
    title(Yname{jn})
end