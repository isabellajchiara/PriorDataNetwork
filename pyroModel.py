import torch
import torn.nn as nn 
import pyro

## create NN

class NeuralNetwork(nn.Module):
    def __init__(self, in_size, out_size):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(in_size, out_size))
        self.bias = nn.Parameter(torch.randn(out_size))
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(),
            nn.ReLU(),
            nn.Linear(),
            nn.ReLU(),
            nn.Linear()
        )
        super().__init__()
        self.weight = nn.Parameter(torch.randn(in_size, out_size))
        self.flatten = nn.Flatten()
        self.bias = nn.Parameter(torch.randn(out_size))
        if x is None:
            with pyro.plate("x_plate", seq_len):
                d_ = dist.Normal(torch.tensor([0.0]).to(self.device), torch.tensor([1.0]).to(self.device)).expand(
                    [self.num_features]).to_event(1)
                x = pyro.sample("x", d_)
        out = self.model(x)
        mu = out.squeeze()
        softmax = torch.nn.Softmax(dim=1)
        with pyro.plate("data", out.shape[0]):
            s = softmax(mu)
            obs = pyro.sample('obs', dist.Categorical(probs=s), obs=y).float()

        return x, obs

model = NeuralNetwork(5,2)

assert isinstance(model, nn.Module)
assert not isinstance(model, PyroModule)

## Convert to Pyro

class PyroLinear(NeuralNetwork, PyroModule):
    pass

model = PyroLinear(5, 2)
assert isinstance(model, nn.Module)
assert isinstance(model, NeuralNetwork)
assert isinstance(model, PyroModule)



## parameters

lossFunc = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(),lr=1e-5)

def train(data, model, loss_fn, optimizer):
    size = len(data)
    model.train()

    for batch, (X, y) in enumerate(data):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test(data, model, loss_fn):
    size = len(data)
    num_batches = len(data)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in data:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

epochs = 5
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_dataloader, model, loss_fn, optimizer)
    test(test_dataloader, model, loss_fn)
print("Done!")

## Ccnvert to Pyro

class Bayesian(NeuralNetwork, PyroModule):
    pass

model = Bayesian(5, 2)
assert isinstance(model, nn.Module)
assert isinstance(model, NeuralNetwork)
assert isinstance(model, PyroModule)

>>>>>>> 347948cdcfb1f3ca454975b84ad84a4323836040