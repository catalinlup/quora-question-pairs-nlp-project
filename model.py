import torch
import torch.nn as nn
import torch.optim as optim
import torch.functional as F
from torch.utils.data import DataLoader, TensorDataset



class QuoraNet(nn.Module):

    def __init__(self) -> None:
        super(QuoraNet, self).__init__()
        self.l1 = nn.Linear(18, 30)
        self.activ1 = nn.ReLU()
        self.l2 = nn.Linear(30, 15)
        self.activ2 = nn.ReLU()
        self.l3 = nn.Linear(15, 1)
        self.activ3  = nn.Sigmoid()
      

    def forward(self, x):
        x = self.l1(x)
        x = self.activ1(x)
        x = self.l2(x)
        x = self.activ2(x)
        x = self.l3(x)
        x = self.activ3(x)

        return x


def train_net(net, X_train, y_train, num_epchs, lr=0.1, b_size=500):
    optimizer = optim.Adam(net.parameters(), lr)
    bce_loss = nn.BCELoss()


    losses = []

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=b_size)

    for epoch in range(num_epchs):
        epoch_losses = []
        for i, d in enumerate(train_loader):
            d_in, expected_output = d
            optimizer.zero_grad()
            predicted = net(d_in)

            loss = bce_loss(predicted, expected_output.unsqueeze(1))
            loss.backward()
            optimizer.step()

            losses.append(loss.item())
            epoch_losses.append(loss.item())

            

        print(f'###{epoch + 1} / {num_epchs}###')
        print(f'Loss: {torch.mean(torch.tensor(epoch_losses))}')
    
    return losses
    

def test_net(net, X_test, y_test):
    """
    Evaluates network in batches.

    Args:
        test_loader: Data loader for test set.
        net: Neural network model.
        criterion: Loss function (e.g. cross-entropy loss).
    """

    avg_loss = 0
    correct = 0
    total = 0

    criterion = nn.BCELoss()

    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=10)
    
    # Use torch.no_grad to skip gradient calculation, not needed for evaluation
    with torch.no_grad():
        # iterate through batches
        for data in test_loader:
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # forward pass
            outputs = net(inputs)
            loss = criterion(outputs, labels.unsqueeze(1))

            # keep track of loss and accuracy
            avg_loss += loss
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return avg_loss/len(test_loader), 100 * correct / total    