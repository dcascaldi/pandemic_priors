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
# February/2023
#
##################################################################
using Distributions, LinearAlgebra, TimeSeries, Plots, DataFrames, XLSX, Statistics, ProgressMeter, StatsBase, Dates, SpecialFunctions
@time begin # begin timer
include("./functions_Pandemic_Priors.jl") # Include the auxiliary functions

#######################  Bayesian VAR #######################

#  Define specification of the VAR
constant = 1                   # 0 --> no intercepts, 1 --> with intercepts
nAR = 12                       # Lags of the VAR
nimp = 36                      # horizon of IRFs to be shown
rps = 1000                     # Number of coefficient draws
covid_periods = 6              # Number of COVID-19 periods for time dummies, starting from March/2020; set to zero for no COVID-19 dummies
diff_or_lv = 1                 # 0 --> in differences, 1 --> level
test_stab = 1                  # 0 --> all posterior draws, 1 --> only stationary draws
nshocks = 1                    # Number of identified shocks (Cholesky ordering)
log_vector = [0 1 0 1 1 1 1 0] # Variables in log
bands = [50 16 84]             # Coverage bands
savefigures = 1                # savefigures = 1 --> Save figures as .PNG

# Specify parameters of the Pandemic Priors
λ=0.2                          # overall prior tightness
ϵ=0.001                        # prior for the constant
ϕ=999                          # prior for the pandemic; 999 = optimal, 0.001 = uninformative

####################### Monthly Data #######################

cd(raw"C:\Pandemic Priors") # set your folder path
b1 = DataFrame(XLSX.readtable("Data.xlsx","Sheet1"))
Ynames = ["EBP","S&P 500","Shadow Rate","PCE","PCE Price Index","Employment","Ind. Production","Unemp. Rate"];
time_vec = Date(1975, 1, 1):Month(1):Date(2022, 12, 1)

#############################################################

Yraw= Array{Float64}(b1[:,2:end])
for ee in axes(log_vector,2)
    if log_vector[ee]==1; Yraw[:,ee] = broadcast(log,Yraw[:,ee])*100; end
end
if diff_or_lv ==0
    Yraw_temp = Yraw[2:end,:]
    for ee in axes(log_vector,2)
        if log_vector[ee]==1; Yraw_temp[:,ee] = diff(Yraw[:,ee],dims=1); end
    end
    Yraw = Yraw_temp
end
Traw,n = size(Yraw)
x = mlag2(Yraw,nAR)
if constant==1;  X1 = [x[nAR+1:Traw,:] ones(Traw-nAR,1)]; else; X1 = x[nAR+1:Traw,:]; end

## COVID-19 time dummies
X1 = [X1 zeros(size(X1,1),covid_periods)]
if diff_or_lv ==1
    covid_ind = findfirst(time_vec .== Date(2020, 3, 1))-nAR
else
    covid_ind = findfirst(time_vec .== Date(2020, 3, 1))-nAR-1
end
X1[covid_ind:covid_ind+covid_periods-1,end-covid_periods+1:end] = Matrix{Float64}(I, covid_periods,covid_periods)

## Adjusting matrices
Y1 = Yraw[nAR+1:Traw,:]; T = Traw - nAR; Y = Y1; X = X1; K = size(X,2);

# OLS coefficients
A_OLS = (X'*X)\(X'*Y)
SSE = (Y - X*A_OLS)'*(Y - X*A_OLS); SIGMA_OLS = SSE./(T-K);
RESID_OLS = (Y - X*A_OLS); VCV_OLS = cov(RESID_OLS); A0_OLS = cholesky(VCV_OLS); A0_OLS = A0_OLS.L;

# Priors
if diff_or_lv ==0; δ=0; elseif diff_or_lv ==1 δ=1; end  # prior mean of the coefficient matrix
τ = convert(Float64,10*λ); λ = convert(Float64,λ)
δ = convert(Float64,δ); ϵ = convert(Float64,ϵ)

# Optimal ϕ
ϕ = convert(Float64,ϕ)
if ϕ == 999
    ϕ_temp = [0.001, 0.01, 0.025, 0.050, 0.075, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.75, 1, 2, 5] # grid for ϕ
    ϕTest = zeros(length(ϕ_temp))
    for jj in axes(ϕ_temp,1)
        ϕTest[jj] = OptimalPhi(X,Y,Yraw,nAR,constant,δ,λ,τ,ϵ,ϕ_temp[jj],covid_periods)
        println("Density($(ϕ_temp[jj])) = $(ϕTest[jj])")
    end
    Density, GridPosition = findmax(ϕTest)
    ϕ_use = ϕ_temp[GridPosition] # select the optimal ϕ
    println("Optimal ϕ for the model is $(ϕ_use).")
else
    ϕ_use = ϕ # use pre-defined ϕ
end

# Pandemic Priors and posterior mean
(Xst, Yst, xd, yd) = pandemicpriors(X,Y,Yraw,nAR,constant,δ,λ,τ,ϵ,ϕ_use,covid_periods)
XXst = xd'*xd + X'*X
invXXst = inv(XXst)
XYst = xd'*yd + X'*Y
A_post = invXXst*XYst

RESID = (Yst - Xst*A_post)
SSE_post = RESID'*RESID
inv_SSE_post = Array(Hermitian(inv(SSE_post)))

# Posterior draws
v0=n+2; v1=size(Xst,1)+2-size(Xst,2)
xx=Xst'*Xst; ixx=inv(xx)
chol_ixx = cholesky(Symmetric(ixx)); chol_ixx = chol_ixx.U;
A_companion_T = zeros(rps,n*nAR,n*nAR)
A0hat_T = zeros(rps,n,n)
@showprogress 1 "Drawing coefficients..." for iii=1:rps
        if test_stab ==0
        A_companion_T[iii,:,:],A0hat_T[iii,:,:] = draw_coef_pandemic_priors(SSE_post,v1,chol_ixx,A_post,n,nAR,covid_periods)
    elseif test_stab ==1
        A_companion_T[iii,:,:],A0hat_T[iii,:,:] = draw_coef_pandemic_priors_stab(SSE_post,v1,chol_ixx,A_post,n,nAR,covid_periods)
    end
end

# Calculate IRFs
outs = zeros(rps,nshocks,nimp,n)
@showprogress 1 "Calculating IRFs..." for iii=1:rps
    outs[iii,:,:,:]=do_irfs(A_companion_T[iii,:,:],A0hat_T[iii,:,:],n,nshocks,nimp)
end

# Plots
quantiles=def_quantiles(outs,vec(bands))
for j=1:nshocks
    plot_array = plot_irfs(quantiles[:,j,:,:],nimp,n,Ynames)
    ppp=plot(plot_array...,layout=n,legend=false)
    display(ppp)
    if savefigures ==1; savefig(ppp,"shock_$(lpad(j,1)).pdf"); end
end

end # end timer