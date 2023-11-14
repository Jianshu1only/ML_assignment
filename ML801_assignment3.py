import torch
import torch.nn as nn
import torch.optim as optim

# Define the network architecture
class XNORNet(nn.Module):
    def __init__(self):
        super(XNORNet, self).__init__()
        # Define layers
        self.hidden = nn.Linear(2, 2)  # 2 input neurons, 2 hidden neurons
        self.output = nn.Linear(2, 1)  # 2 hidden neurons, 1 output neuron
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.sigmoid(self.hidden(x))
        x = self.sigmoid(self.output(x))
        return x

# Initialize the network
model = XNORNet()

# Define loss function and optimizer
criterion = nn.BCELoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

# Training data and labels for XNOR
data = torch.tensor([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])
labels = torch.tensor([[1.], [0.], [0.], [1.]])

# Training loop
for epoch in range(1000):  # Number of epochs
    optimizer.zero_grad()
    # Forward pass
    outputs = model(data)
    loss = criterion(outputs, labels)
    # Backward pass and optimization
    loss.backward()
    optimizer.step()

    if (epoch+1) % 100 == 0:
        print(f'Epoch [{epoch+1}/1000], Loss: {loss.item():.4f}')

# Test the network
with torch.no_grad():
    predicted = model(data).round()
    print("Predicted binary outputs for XNOR operation:")
    print(predicted)
