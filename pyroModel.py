class BayesianModel(PyroModule):
    def __init__(self, in_size, out_size):
       super().__init__()
       self.bias = PyroSample( #PyroSample makes Bayesian
           prior=dist.LogNormal(0, 1).expand([out_size]).to_event(1))
       self.weight = PyroSample(
           prior=dist.Normal(0, 1).expand([in_size, out_size]).to_event(2))

    def forward(self, input):
        return self.bias + input @ self.weight  # samples bias and weight
