import torch
import torn.nn as nn 
import pyro

## create NN

class BayesianNetwork(nn.Module):
    def __init__(self, model_spec, device='cuda'):
        super().__init__()

        self.device = device
        self.num_features = modelspec['num_features']

        mu, sigma = torch.tensor([0.0]).to(self.device), torch.tensor([1.0]).to(self.device)

        self.fc1 = PyroModule[nn.Linear](self.num_features, model_spec['embed'])
        self.fc1.weight = PyroSample(
            dist.Normal(mu,sigma).expand([model_spec['embed'], self.num_features]).to_event(2))
        self.fc1.bias = PyroSample(
            dist.Normal(mu,sigma).expand([model_spec['embed']].to_event(1)))

        self.fc2 = PyroModule[nn.Linear](self.num_features, model_spec['embed'])
        self.fc2.weight = PyroSample(
            dist.Normal(mu,sigma).expand([2,model_spec['embed']]).to_event(2))
        self.fc2.bias = PyroSample(
            dist.Normal(mu,sigma).expand([2].to_event(1)))

        self.model = torch.nn.Sequential(self.fc1,self.fc2)

        self.to(self.device)

    def forward(self, x=None, y=None, seq_len=1):
        if x is None:
            with pyro.plate("x_plant", seq_len):
                d_ = dist.Normal(torch.tensor([0.0]).to(self.device), 
                                 torch.tensor([1.0]).to(self.device)).expand([self.num_features]).to_event(1)
                x = pyro.sample("x",d_)

        out = self.model(x)
        mu = out.squeeze()
        softmax = torch.nn.Softmax(dim=1)
        with pyro.plate("data",out.shape[0]):
            s = softmax(mu)
            obs = pyro.sample('obs',dist.Categorical(probs=s), obs=y).float()

        return x, obs

                
                                 
