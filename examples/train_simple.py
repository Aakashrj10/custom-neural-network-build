# examples/train_simple.py

from tensor import Tensor
from nn.linear import Linear
from nn.activations import ReLU
from optim.sdg import SDG

class MLP(Module):
    def __init__(self):
        self.l1 = Linear(2, 16)
        self.l2 = Linear(16, 1)
        self.relu = ReLU()

    def __call__(self, x):
        return self.l2(self.relu(self.l1(x)))
    

# dummy XOR data
X = Tensor([[0,0],[0,1],[1,0],[1,1]], requires_grad=False)
y = Tensor([[0],[1],[1],[0]], requires_grad=False)

model = MLP()
opt = SGD(model.parameters(), lr=0.1)

for epoch in range(5000):
    pred = model(X)
    loss = ((pred - y)** 2).mean()
    
    opt.zero_grad()
    loss.backward()
    opt.step()

    if epoch % 500 == 0:
        print("epoch:", epoch, "loss:", loss.data)