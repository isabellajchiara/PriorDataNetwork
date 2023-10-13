
# Create NN

class NeuralNetwork(nn.Module):
    def __init__(self, in_size, out_size):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(in_size, out_size))
        self.flatten = nn.Flatten()
        self.bias = nn.Parameter(torch.randn(out_size))

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

model = NeuralNetwork(5,2)

assert isinstance(model, nn.Module)
assert not isinstance(model, PyroModule)

## Convert to Pyro

class Bayesian(NeuralNetwork, PyroModule):
    pass

model = Bayesian(5, 2)
assert isinstance(model, nn.Module)
assert isinstance(model, NeuralNetwork)
assert isinstance(model, PyroModule)
