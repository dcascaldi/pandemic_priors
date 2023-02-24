# Associate functions for the Pandemic Priors
# Created by Danilo Cascaldi-Garcia
# Federal Reserve Board

using Distributed, Distributions

@everywhere begin
    
function mlag2(X::Array{Float64,2},p::Int64)
    # Created by Danilo Cascaldi-Garcia
    Traw,N =size(X)
    Xlag=zeros(Traw,N*p)
    for ii=1:p
        Xlag[p+1:Traw,(N*(ii-1)+1):N*ii]=X[p+1-ii:Traw-ii,1:N]
    end
   return Xlag
end 

function mlag1(X::Array{Float64,1},p::Int64)
    # Created by Danilo Cascaldi-Garcia
    Traw =size(X,1)
    Xlag=zeros(Traw,p)
    for ii=1:p
        Xlag[p+1:Traw,((ii-1)+1):ii]=X[p+1-ii:Traw-ii]
    end
   return Xlag
end

function pandemicpriors(X::Array{Float64,2},Y::Array{Float64,2},Yraw::Array{Float64,2},nAR::Int64,constant::Int64,δ::Float64,λ::Float64,τ::Float64,ϵ::Float64,ϕ::Float64,covid_periods::Int64)
    # Created by Danilo Cascaldi-Garcia
    T = size(X,1)
    Traw = T+nAR
    p=nAR
    mu=mean(Y,dims=1)
    n=size(Y,2)
    # Construct a univariate autoregressive model of order p for each n
    Coef = Array{Float64}(undef,p+1,n)
    ResidAR = Array{Float64}(undef,Traw-nAR,n)
    for i=1:n
        YlagAR = mlag1(Yraw[:,i],p);
        X_AR = [YlagAR[nAR+1:Traw,:] ones(Traw-nAR,1)]
        Y_AR = Yraw[nAR+1:Traw,i]
        A_AR = (X_AR'*X_AR)\(X_AR'*Y_AR)
        Coef[:,i] = A_AR
        ResidAR[:,i] = Y_AR - X_AR*Coef[:,i]
    end

    SSE2 = (ResidAR)'*(ResidAR)
    sizeAR = size(ResidAR,1)
    σ_AR = SSE2./sizeAR
    σ = zeros(size(σ_AR))
    for ll in axes(σ_AR,1)
        σ[ll,ll] = sqrt(σ_AR[ll,ll])
    end
    if λ>0
        bb = diagm(diag(σ.*δ)./λ)
        dd = diagm(diag(σ))
        ff = diagm(diag(σ)./λ)  
        jp=diagm([1:1:p;])
        if constant == 1
            yd1=[bb; zeros(n*(p-1),n); dd; zeros(covid_periods+1,n)]
            xd1=[kron(jp,ff) zeros(n*p,covid_periods+1); zeros(n,n*p+covid_periods+1)
                zeros(1,n*p) ϵ zeros(1,covid_periods)
                zeros(covid_periods,n*p+1) diagm(ones(covid_periods))*ϕ]  
        else
            yd1=[bb; zeros(n*(p-1),n); dd; zeros(covid_periods,n)]
            xd1=[kron(jp,ff) zeros((n*p),covid_periods); zeros(n,(n*p)+covid_periods)
                zeros(covid_periods,n*p) diagm(ones(covid_periods))*ϕ] 
        end              
        if τ>0
            bb2 = diagm(vec((δ.*mu)./τ))
            yd2=bb2;
            if constant == 1
                xd2=[kron(ones(1,p),yd2) zeros(n,covid_periods+1)]
            else
                xd2=[kron(ones(1,p),yd2) zeros(n,covid_periods)]
            end
        else
            yd2 = []
            xd2 = []               
        end
    end
    yd=[yd1;yd2]
    xd=[xd1;xd2]
    Yst = [Y; yd]
    Xst = [X; xd]
    return Xst, Yst, xd, yd;
end

function draw_coef_pandemic_priors_stab(SSE_post::Array{Float64,2},v1::Int64,chol_ixx::UpperTriangular{Float64,Array{Float64,2}},A_post::Array{Float64,2},n::Int64,nAR::Int64,covid_periods::Int64)
    # Created by Danilo Cascaldi-Garcia
    control2=0;
    while control2==0
        σrep = rand(InverseWishart(v1,SSE_post)) # draw σ
        chol_σrep = cholesky(σrep)
        nbeta_dr=A_post+(chol_ixx')*(randn((n*nAR+covid_periods+1),n))*(chol_σrep.U)
        A0hat=chol_σrep.L
        A_companion_dr=zeros(n*nAR,n*nAR)
        A_companion_dr[1:n,:]=nbeta_dr[1:n*nAR,:]'
        A_companion_dr[n+1:n*nAR,1:n*nAR-n]=Matrix{Float64}(I, n*nAR-n, n*nAR-n)
        EI_a=eigen(A_companion_dr); eigAR_a=maximum(broadcast(abs,EI_a.values));
        if eigAR_a<1.01
            control2=1
            return A_companion_dr, A0hat
        else
            continue
        end
    end 
end

function draw_coef_pandemic_priors(SSE_post::Array{Float64,2},v1::Int64,chol_ixx::UpperTriangular{Float64,Array{Float64,2}},A_post::Array{Float64,2},n::Int64,nAR::Int64,covid_periods::Int64)
    # Created by Danilo Cascaldi-Garcia
    σrep = rand(InverseWishart(v1,SSE_post)) # draw σ
    chol_σrep = cholesky(σrep)
    nbeta_dr=A_post+(chol_ixx')*(randn((n*nAR+covid_periods+1),n))*(chol_σrep.U)
    A0hat=chol_σrep.L
    A_companion_dr=zeros(n*nAR,n*nAR)
    A_companion_dr[1:n,:]=nbeta_dr[1:n*nAR,:]'
    A_companion_dr[n+1:n*nAR,1:n*nAR-n]=Matrix{Float64}(I, n*nAR-n, n*nAR-n)
    return A_companion_dr, A0hat
end

function do_irfs(A_companion_dr::Array{Float64,2},IMP_dr::Array{Float64,2},n::Int64,nshocks::Int64,nimp::Int64)
    # Created by Danilo Cascaldi-Garcia
    U1_dr=[IMP_dr; zeros(n*nAR-n,n)]
    nnn_dr = size(U1_dr,1)
    Eye_comp_dr = Matrix{Float64}(I, nnn_dr, nnn_dr)-A_companion_dr
    Zk1_dr = zeros(nshocks,nimp,n*nAR)
    impulse1_dr = zeros(nshocks,nimp,n)
    for r=1:nshocks
        for k=1:nimp
            Zk1_dr[r,k,:]=(A_companion_dr^(k-1)*U1_dr[:,r])'
        end
        impulse1_dr[r,:,1:end]=Zk1_dr[r,:,1:n]
        if impulse1_dr[r,1,r]<0; impulse1_dr[r,:,:] = impulse1_dr[r,:,:]*(-1); end
    end
    return impulse1_dr;
end

function def_quantiles(out::Array{Float64,4},prob::Array{Int64,1})
    # Created by Danilo Cascaldi-Garcia
    shocks = size(out,2); hor=size(out,3); var_ef = size(out,4)
    nprob=size(prob,1)
    quantiles_out=zeros(nprob,shocks,hor,var_ef)  
    for g=1:nprob
        for i=1:shocks
            for h=1:hor
                for j=1:var_ef
                    quantiles_out[g,i,h,j]=percentile(out[:,i,h,j],prob[g])
                end
            end
        end
    end
    return quantiles_out
end

function plot_irfs(quantiles::Array{Float64,3},nimp::Int64,n::Int64,Ynames::Array{String,1})
    # Created by Danilo Cascaldi-Garcia
    plot_array = Any[]
    for i=1:n
        plot(1:nimp,zeros(nimp),s=:dot,c=:black)
        plot!(1:nimp,quantiles[2,:,i],fillrange = quantiles[3,:,i],fillalpha =0.35,linealpha=0,c=:black)
        ppp=plot!(1:nimp,quantiles[1,:,i],c=:black,title=Ynames[i],titlefont=font(10))
        push!(plot_array,ppp)
    end
    return plot_array;
end

function OptimalPhi(X::Array{Float64,2},Y::Array{Float64,2},Yraw::Array{Float64,2},nAR::Int64,constant::Int64,δ::Float64,λ::Float64,τ::Float64,ϵ::Float64,ϕ::Float64,covid_periods::Int64)
    # Created by Danilo Cascaldi-Garcia
    _, _, xd, yd = pandemicpriors(X, Y, Yraw, nAR, constant, δ, λ, τ, ϵ, ϕ, covid_periods)
    T = size(X,1)
    n = size(Y,2)
    # Dummies
    xx0 = xd' * xd
    ixx0 = inv(xx0)
    b0 = ixx0 * xd' * yd
    v0 = n + 2
    e0 = yd - xd * b0
    σ0 = e0' * e0   
    # Posterior
    v1 = v0 + T   
    # Density
    aPP = Matrix{Float64}(I, T, T) + X * ixx0 * X'
    PP = inv(aPP)
    QQ = σ0   
    # Log n-variate Gamma function
    # r1 
    x_log = ones(n,1)*(v0/2) + (ones(n,1)-[1:1:n;])/2
    sum_log = sum(broadcast(loggamma,x_log), dims=1)
    r1 = n * (n - 1) / 4 * log(pi) + sum_log[1] 
    # r2
    x_log = ones(n,1)*(v1/2) + (ones(n,1)-[1:1:n;])/2
    sum_log = sum(broadcast(loggamma,x_log), dims=1)
    r2 = n * (n - 1) / 4 * log(pi) + sum_log[1]
    # p(Y)
    py = -(T*n/2)*log(pi) + (1/2)*n * log(det(PP)) + (v0/2)*log(det(QQ)) + (r2-r1) - (v1/2)*log(det(QQ + (Y - X*b0)' * PP * (Y - X*b0)))
    return py
end
end