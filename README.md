# pandemic_priors
A simple, easy, and flexible way of estimating Bayesian VARs taking into consideration the pandemic period, as a Minnesota prior with time dummies, available in MATLAB and Julia.  Codes embed a test for the optimal level of shrinkage for the pandemic period.

Paper available at my website: www.danilocascaldigarcia.com

Use of code for research purposes is permitted as long as proper reference to source is given.

This version: February 2023

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

MATLAB:

The main program BVAR_Pandemic_Priors.m performs the Pandemic Priors Bayesian VAR estimation with the time dummies on March to August 2020, and identifies an EBP shock with a recursive Cholesky structure, where EBP is ordered first.

"covid_periods" defines how many monthly dummies to include (starting from and including March 2020).  Set to zero to run a conventional Minnesota Prior as in Banbura, Giannone, and Reichlin (2010).

"phi" defines how much signal the econometrician would like to take from the pandemic period.  With phi = 999, the value for phi will be the optimal from a marginal likelihood standpoint. With phi close to zero the time dummies are "active," soaking all the pandemics variance; with phi close to infinity the time dummies are "inactive," and the model boils down to a conventional Minnesota Prior.

Dummy observations are created with the auxiliary function pandemicpriors.m.

Optimal phi calculated with the auxiliary function OptimalPhi.m.

Codes written in MATLAB 2022a

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

Julia:

The main program BVAR_Pandemic_Priors.jl performs the Pandemic Priors Bayesian VAR estimation with the time dummies on March to August 2020, and identifies an EBP shock with a recursive Cholesky structure, where EBP is ordered first.

"covid_periods" defines how many monthly dummies to include (starting from and including March 2020).  Set to zero to run a conventional Minnesota Prior as in Banbura, Giannone, and Reichlin (2010).

"\phi" defines how much signal the econometrician would like to take from the pandemic period. With \phi =999, the value for \phi will be the optimal from a marginal likelihood standpoint. With \phi close to zero the time dummies are "active," soaking all the pandemics variance; with phi close to infinity the time dummies are "inactive," and the model boils down to a conventional Minnesota Prior.

Auxiliary functions, including the creation of the dummy observations are stored in functions_Pandemic_Priors.jl.

Codes written in Julia v1.8.5
