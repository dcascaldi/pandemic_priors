%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% This code implements the Pandemic Priors of Cascaldi-Garcia (20232)
% with the Asymmetric Conjugate Priors of Chan (2022)
%
% This code is free to use for academic purposes only, provided that the 
% papers are cited as:
%
% Cascaldi-Garcia, D. (2022). Pandemic priors.
% International Finance Discussion Paper, (1352).
%
% and
%
% Chan, J. C. (2022). Asymmetric conjugate priors for large Bayesian VARs.
% Quantitative economics, 13(3), 1145-1169.
%
% This code comes without technical support of any kind. It is expected to
% reproduce the results reported in the paper. Under no circumstances will
% the authors be held responsible for any use (or misuse) of this code in
% any way.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

set(0,'defaulttextinterpreter','latex')
set(0,'defaultLegendInterpreter','latex');
set(0,'defaultAxesTickLabelInterpreter','latex');

clear; clc;
close all
addpath('./utility');

tic
p = 12;         % lags in the VAR
nsim = 10000;   % # of posterior draws
horizon = 36;   % # of steps for impulse responses

% load data
log_vector = [0 1 0 1 1 1 1 0]; % Variables in log
covid_periods = 6;              % Number of COVID-19 periods for time dummmies, starting from March/2020; set to zero for no COVID-19 dummies
diff_or_lv = 1;                 % 0 --> in differences, 1 --> level

%% Read data
data = readmatrix("Data.xlsx");
data = data(:,2:end);
Yname = {'EBP','S\&P 500','Shadow Rate','PCE','PCE Price Index','Employment','Ind. Production','Unemp. Rate'};
time_vec = datetime(1975,1,1):calmonths(1):datetime(2022,12,1); % Specify the dates vector
%% Preparing the data
Yraw = data;
for ee=1:size(log_vector,2)
    if log_vector(ee)==1; Yraw(:,ee) = log(Yraw(:,ee))*100; end
end
if diff_or_lv ==0
    Yraw_temp = Yraw(2:end,:);
    for ee=1:size(log_vector,2)
        if log_vector(ee)==1; Yraw_temp(:,ee) = diff(Yraw(:,ee)); end
    end
    Yraw = Yraw_temp;
end

var_id = 1:size(log_vector,2); % Variables    
idx_ns = 1:size(log_vector,2); % Index for variables in levels

Y0 = data(1:p,var_id);  % save the first p obs as the initial conditions
Y = data(p+1:end,var_id);
[T,n] = size(Y);
tmpY = [Y0(end-p+1:end,:); Y];
Z = zeros(T,n*p); 
for ii=1:p
    Z(:,(ii-1)*n+1:ii*n) = tmpY(p-ii+1:end-ii,:);
end
Z = [ones(T,1) Z];

%% Asymmetric Conjugate Priors - Chan (2022)

% find the optimal kappa values
[~,kappa] = get_OptKappa(Y0,Y,Z,p,[.04,.0016],'redu',idx_ns);
sig2 = get_resid_var(Y0,Y);
prior_redu = prior_ACP_redu(n,p,kappa,sig2,idx_ns);

%  sample nbatch draws from the posterior
[store_alp,store_beta,store_Sig] = sample_ThetaSig(Y0,Y,p,prior_redu,nsim);
% obtain the reduced-form parameters
[store_Btilde,store_Sigtilde] = getReducedForm(store_alp,store_beta,store_Sig);

C_auto_T_base = zeros(nsim,n,n);
Btilde = NaN(nsim,n*p+1,n);
for iii=1:nsim
    Btemp = reshape(store_Btilde(iii,:),n*p+1,n);
    Btilde(iii,:,:) = Btemp;
    C_auto_T_base(iii,:,:) = Btemp(2:n+1,1:n);
end

%% Pandemic Priors - Cascaldi-Garcia (2022)

% COVID-19 dummies
if diff_or_lv ==1
    %covid_ind = datefind(datetime(2020,3,1),time_vec,1)-p;
    covid_ind = find(datetime(2020,3,1)==time_vec)-p; % Find March/2020
else
    %covid_ind = datefind(datetime(2020,3,1),time_vec,1)-p-1;
    covid_ind = find(datetime(2020,3,1)==time_vec)-p-1; % Find March/2020
end
Zpp = [Z, zeros(size(Z,1),covid_periods)];
Zpp(covid_ind:covid_ind+covid_periods-1,end-covid_periods+1:end) = eye(covid_periods);

% find the optimal kappa and phi values
warning('off','all')
[ml_opt,kappa_pp] = get_OptKappa_pp(Y0,Y,Zpp,p,[.04,.0016,0.2],'redu',idx_ns,covid_periods); % 0.2 initial value for phi
sig2 = get_resid_var(Y0,Y);
prior_redu = prior_ACP_redu_pp(n,p,kappa_pp,sig2,idx_ns,covid_periods);

%  sample nbatch draws from the posterior
[store_alp_pp,store_beta_pp,store_Sig_pp] = sample_ThetaSig_pp(Y0,Y,Zpp,p,prior_redu,nsim,covid_periods);
% obtain the reduced-form parameters
[store_Btilde_pp,store_Sigtilde_pp] = getReducedForm(store_alp_pp,store_beta_pp,store_Sig_pp);

C_auto_T = zeros(nsim,n,n);
Btilde_pp = NaN(nsim,n*p+1+covid_periods,n);
for iii=1:nsim
    Btemp = reshape(store_Btilde_pp(iii,:),n*p+1+covid_periods,n);
    Btilde_pp(iii,:,:) = Btemp;
    C_auto_T(iii,:,:) = Btemp(2:n+1,1:n);
end

    %% Histogram of the auto regressive coefficients
    fontsize = 12;
    fontlegend = fontsize - 4;
    SIZE_VARS = size(Yname,2);
    p_lines = floor(SIZE_VARS/3);
    p_cols = ceil(SIZE_VARS/p_lines);
    hist_bins = 20;
    ss = [];
    
    h=figure('Units','normalized','Color',[0.9412 0.9412 0.9412],'outerposition',[0,0,0.6,1],'Name','hist_auto');
    figure(h);
    for uuu=1:n
        subplot(4,2,uuu)
        ss(1) = histogram(squeeze(C_auto_T_base(:,uuu,uuu)),hist_bins,'EdgeColor',[0/255 128/255 255/255],'FaceColor',[0/255 178/255 255/255]); hold on
        ss(2) = histogram(squeeze(C_auto_T(:,uuu,uuu)),hist_bins,'EdgeColor',[255/255 100/255 100/255],'FaceColor',[255/255 150/255 150/255]);
        set(gca,'FontSize',fontlegend)
        title(Yname(uuu),'FontSize',fontsize) %'FontWeight','bold',
        hold off;
        if uuu==1
            legend([ss(2) ss(1)],'Pandemic Priors','Asymmetric priors','Location','northwest','FontSize',fontlegend); legend boxoff;
        end
    end
    set(gca,'Box','on');

toc
