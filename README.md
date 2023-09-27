Prior data fitted models are can predict on unseen data regardless of population structure

First we take our population data and simulate an unstructure population from our highly structure Elite Breeding Lines.
This way, we train our model on priors that are not hyper-specific to our breeding population

The we create our PFNN which approximates the posterior predictive distribution where factors like strucutre and recombination are implicit.
