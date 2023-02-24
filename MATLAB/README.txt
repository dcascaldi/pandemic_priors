Files and replication MATLAB codes of Cascaldi-Garcia, D., "Pandemic Priors"

Use of code for research purposes is permitted as long as proper reference to source is given.
______________________________________

The main program BVAR_Pandemic_Priors.m performs the Pandemic Priors Bayesian VAR estimation with the time dummies on March to August 2020, and identifies an EBP shock with a recursive Cholesky structure, where EBP is ordered first.  BVAR_Pandemic_Priors_parallel.m is the version with parallelized processes for draws and IRFs.

"covid_periods" defines how many monthly dummies to include (starting from and including March 2020).  Set to zero to run a conventional Minnesota Prior as in Banbura, Giannone, and Reichlin (2010).

"phi" defines how much signal the econometrician would like to take from the pandemic period.  With phi = 999, the value for phi will be the optimal from a marginal likelihood standpoint. With phi close to zero the time dummies are "active," soaking all the pandemics variance; with phi close to infinity the time dummies are "inactive," and the model boils down to a conventional Minnesota Prior.

Dummy observations are created with the auxiliary function pandemicpriors.m.

Optimal phi calculated with the auxiliary function OptimalPhi.m.

Codes written in MATLAB 2022a

This version: February 2023

Danilo Cascaldi-Garcia
