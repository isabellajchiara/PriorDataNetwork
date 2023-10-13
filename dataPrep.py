import os
import torch
import torch.nn as nn
import pyro
import pyro.distributions as dist
import pyro.poutine as poutine
from torch.distributions import constraints
from pyro.nn import PyroModule, PyroParam, PyroSample
from pyro.nn.module import to_pyro_module_
from pyro.infer import SVI, Trace_ELBO
from pyro.infer.autoguide import AutoNormal
from pyro.optim import Adam
import pyro.distributions as dist
import pyro.distributions.constraints as constraints

vars = pd.read_csv("unstructuredGeno.csv")
resp = pd.read_csv("unstructuredPheno.csv")
data = resp.join(vars)

train = torch.tensor(data.values, dtype=torch.float)
x = train[:,0]
y = train[:,1:3548]


for X, y in data
    print(f"Shape of X: {X.shape}")
    print(f"Shape of y: {y.shape} {y.dtype}")
    break
