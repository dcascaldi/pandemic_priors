Files and replication Python codes of Cascaldi-Garcia, D., "Pandemic Priors"

Use of code for research purposes is permitted as long as proper reference to source is given.
______________________________________

The main program BVAR_Pandemic_Priors.py or BVAR_Pandemic_Priors.ipynb (Jupyter Notebook) performs the Pandemic Priors Bayesian VAR estimation with the time dummies on March to August 2020, and identifies an EBP shock with a recursive Cholesky structure, where EBP is ordered first.

"covid_periods" defines how many monthly dummies to include (starting from and including March 2020).  Set to zero to run a conventional Minnesota Prior as in Banbura, Giannone, and Reichlin (2010).

"phi" defines how much signal the econometrician would like to take from the pandemic period. With phi =999, the value for phi will be the optimal from a marginal likelihood standpoint. With phi close to zero the time dummies are "active," soaking all the pandemics variance; with phi close to infinity the time dummies are "inactive," and the model boils down to a conventional Minnesota Prior.

Auxiliary functions, including the creation of the dummy observations are stored in functions_Pandemic_Priors.py.

Codes written in Python 3.9.13 64-bit

This version: March 2023

Danilo Cascaldi-Garcia
