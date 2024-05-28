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
% February/2023
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
tic
set(0,'defaulttextinterpreter','latex');set(0,'defaultLegendInterpreter','latex');set(0,'defaultAxesTickLabelInterpreter','latex');
clear; close all;

% Define specification of the BVAR
constant = 1;                   % 1 --> add intercepts, 0 --> no intercepts
nAR = 12;                       % Lags of the VAR
nimp = 36;                      % IRF horizon
rps = 10000;                     % Number of coefficient draws
covid_periods = 6;              % Number of COVID-19 periods for time dummmies, starting from March/2020; set to zero for no COVID-19 dummies
diff_or_lv = 1;                 % 0 --> in differences, 1 --> level
test_stab = 1;                  % 0 --> all posterior draws, 1 --> only stationary draws
nshocks = 1;                    % Number of identified shocks (Cholesky ordering)
log_vector = [0 1 0 1 1 1 1 0]; % Variables in log
bands = [50 16 84];             % Coverage bands
savefigures = 0;                % savefigures = 1 --> Save figures as .PNG

% Specify parameters of the Pandemic Priors
lambda = 0.2;                   % degree of overall prior tightness
epsilon = 0.001;                % prior for the constant
phi=    999;                    % prior for the pandemic; 999 = optimal, 0.001 = uninformative

%% Read data
data = readmatrix("Data.xlsx");
data = data(:,2:end);
Yname = {'EBP','S\&P 500','Shadow Rate','PCE','PCE Price Index','Employment','Ind. Production','Unemp. Rate'};
time_vec = datetime(1975,1,1):calmonths(1):datetime(2022,12,1); % Specify the dates vector

%% Preparing the data
n_cores = feature('numcores');
fprintf('Working with %.0f parallel cores\n\n',n_cores);

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
[Traw, nvar] = size(Yraw); Y1 = Yraw;
for ii=1:nAR; Ylag(nAR+1:Traw,(nvar*(ii-1)+1):nvar*ii)=Y1(nAR+1-ii:Traw-ii,1:nvar); end
if constant==1; X1 = [Ylag(nAR+1:Traw,:) ones(Traw-nAR,1)]; else; X1 = Ylag(nAR+1:Traw,:); end

%% COVID-19 time dummies
X1 = [X1, zeros(size(X1,1),covid_periods)];
if diff_or_lv ==1
    %covid_ind = datefind(datetime(2020,3,1),time_vec,1)-nAR;
    covid_ind = find(datetime(2020,3,1)==time_vec)-nAR; % Find March/2020
else
    %covid_ind = datefind(datetime(2020,3,1),time_vec,1)-nAR-1;
    covid_ind = find(datetime(2020,3,1)==time_vec)-nAR-1; % Find March/2020
end
X1(covid_ind:covid_ind+covid_periods-1,end-covid_periods+1:end) = eye(covid_periods);

%% Finalizing data
[Traw3, K] = size(X1); Y1 = Y1(nAR+1:Traw,:); T = Traw - nAR; Y = Y1; X = X1;

%% Dummy observation priors and posterior mean
if nAR>1; mus=mean(Yraw(1:nAR,:)); elseif nAR==1; mus=Yraw(1,:); end
tau = 10*lambda;
if diff_or_lv ==0; delta=0; elseif diff_or_lv ==1; delta=1; end         % prior mean of the coefficient matrix
A_OLS = (X'*X)\(X'*Y); a_OLS = A_OLS(:);
SSE = (Y - X*A_OLS)'*(Y - X*A_OLS); SIGMA_OLS = SSE./(T-K);
RESID_OLS = (Y - X*A_OLS); VCV_OLS=cov(RESID_OLS,1);

if phi ==999 % Optimal phi
    phi_temp = [0.001, 0.01, 0.025, 0.050, 0.075, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40,...
        0.45, 0.50, 0.75, 1, 2, 5]; % Grid for phi
    PhiTest = zeros(length(phi_temp),1);
    for jj=1:length(phi_temp)
        PhiTest(jj)=OptimalPhi(X,Y,Yraw,nAR,constant,delta,lambda,tau,epsilon,phi_temp(jj),covid_periods);
        fprintf('Density(%.3f) = %g\n',phi_temp(jj),PhiTest(jj));
    end
    [Density, GridPosition] = max(PhiTest);
    phi_use = phi_temp(GridPosition);
    fprintf('Optimal phi for the model is %g.\n\n',phi_use);
else
    phi_use = phi;
end
%% Pandemic Priors and posterior mean
[Xst, Yst, xd, yd] = pandemicpriors(X,Y,Yraw,nAR,constant,delta,lambda,tau,epsilon,phi_use,covid_periods);
XXst = xd'*xd + X'*X;
invXXst = (XXst)\eye(size(XXst,2)); % inv(XXst)
XYst = xd'*yd + X'*Y;
V_post = invXXst; A_post = invXXst*XYst; a_post = A_post(:);

if constant == 1
    Mhat=A_post(1:end-1,:)'; M_OLS = A_OLS(1:end-1,:)'; RESID = (Yst - Xst*A_post);
    SSE_post = RESID'*RESID; VCV=cov(RESID,1);
elseif constant == 0
    Mhat=A_post(:,:)'; M_OLS = A_OLS(:,:)'; RESID = (Yst - Xst*A_post);
    SSE_post = RESID'*RESID; VCV=cov(RESID,1);
end

%% Posterior draws
discarded = 0; jj=1; T=size(Xst,1); [T_tot,m_tot] = size(data);
v0=m_tot+2; v1=size(Xst,1)+2-size(Xst,2);
mstar = A_post(:); xx=Xst'*Xst;
ixx=xx\eye(size(xx,2));  %inv(X0'X0)

A_companion_T = zeros(rps,nvar*nAR,nvar*nAR);
A0hat_T = zeros(rps,nvar,nvar);
parfor iii=1:rps
    if iii ==1
        fprintf('Starting %.0f posterior draws with %.0f threads\n',rps,n_cores);
    elseif iii==rps
        fprintf("Posterior draws finished!\n\n")
    end
    control2=0;
    while control2==0
        sigmarep = iwishrnd(SSE_post,v1); % draw SIGMA
        nbeta_dr=A_post+(chol(ixx)')*(randn((nvar*nAR+covid_periods+constant),nvar))*(chol(sigmarep));
        A0hat=chol(sigmarep)';
        A_companion_dr=zeros(nvar*nAR,nvar*nAR);
        A_companion_dr(1:nvar,:)=nbeta_dr(1:nvar*nAR,:)';
        A_companion_dr(nvar+1:nvar*nAR,1:nvar*nAR-nvar)=eye(nvar*nAR-nvar);
        EI_a=eig(A_companion_dr); ROOTS_a=abs(EI_a); ROOTS_a = sortrows(ROOTS_a,1);
        eigAR_a = ROOTS_a(nAR*nvar,:);
        if test_stab ==1
            if eigAR_a<1.01 % Check stability
                control2=1;
            else
                discarded = discarded+1;
                continue
            end
        else
            control2=1;
        end
    end
    A_companion_T(iii,:,:) = A_companion_dr;
    A0hat_T(iii,:,:) = A0hat;
end

%% IRFs
outs = zeros(nshocks,rps,nimp,nvar);
parfor iii=1:rps
    if iii ==1
        fprintf('Starting %.0f IRFs with %.0f threads\n',rps,n_cores);
    elseif iii==rps
        fprintf("IRFs finished!\n\n")
    end
    IMP_dr=squeeze(A0hat_T(iii,:,:));
    A_companion_dr=squeeze(A_companion_T(iii,:,:));
    invIMP_dr = (IMP_dr)\eye(size(IMP_dr,2)); % inv(IMP_dr)
    U1_dr=[IMP_dr; zeros(nvar*nAR-nvar,nvar)];
    nnn_dr = size(U1_dr,1);
    Eye_comp_dr = eye(nnn_dr)-A_companion_dr;
    invEye_comp_dr = (Eye_comp_dr)\eye(size(Eye_comp_dr,2)); % inv(Eye_comp)
    Zk1_dr = zeros(nvar,nimp,nvar*nAR);
    impulse1_dr = zeros(3,nimp,nvar);
    for r=1:nshocks
        for k=1:nimp
            Zk1_dr(r,k,:)=(A_companion_dr^(k-1)*U1_dr(:,r))';
        end
        impulse1_dr(r,:,1:end)=Zk1_dr(r,:,1:nvar);
        if impulse1_dr(r,1,r)<0; impulse1_dr(r,:,:) = impulse1_dr(r,:,:)*(-1);end
        for ss=1:nvar
            outs(r,iii,:,ss)=impulse1_dr(r,:,ss);
        end
    end
end

%% Plot figures
for pp=1:nshocks
    SIZE_VARS = size(Yname,2); p_lines = floor(SIZE_VARS/3); p_cols = ceil(SIZE_VARS/p_lines); selec = 1:nshocks; xaxis = 1:nimp; xaxis = xaxis';
    h=figure('Units','normalized','Color',[0.9412 0.9412 0.9412],'outerposition',[0,0,0.8,0.8],'Name',eval(['''' cell2mat(Yname(selec(pp))) '-shock''']));
    figure(h);
    for uuu=1:nvar
        subplot(p_lines,p_cols,uuu)
        temp=squeeze(outs(pp,:,:,uuu)); temp1=squeeze(prctile(temp,bands,1))';
        grpyat = [(1:nimp)' temp1(1:nimp,2); (nimp:-1:1)' temp1(nimp:-1:1,3)];
        patch(grpyat(:,1),grpyat(:,2),[0.7 0.7 0.7],'edgecolor',[0.65 0.65 0.65]);  hold on;
        ss(1) = plot(xaxis,temp1(1:nimp,1),'k-','LineWidth',3);
        plot(xaxis,zeros(nimp),':k');
        set(gca,'FontSize',16)
        title(Yname(uuu),'FontSize',16) %'FontWeight','bold',
        ylabel('percent','FontSize',16)
        xlabel('months','FontSize',16)
        set(gca,'XTick',0:4:nimp)
        axis tight
        hold off;
    end
end

%% Save figures
if savefigures ==1
    h = get(0,'children'); h = sort(h);
    for i=1:length(h); saveas(h(i), get(h(i),'Name'), 'png'); end
end
toc
