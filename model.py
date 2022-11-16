from torch import nn
import numpy as np
import torch


# Simple Multi-Layer Perceptron applied with Monte-Carlo Dropout
class SimpleMLP(nn.Module):
    def __init__(self, input_units=1, dropout_rate=0, hidden_units=256):
        super(SimpleMLP, self).__init__()

        # Setting up model parameters
        self.hidden_units = hidden_units
        self.input_units = input_units
        self.dropout_rate = dropout_rate

        # Defining MLP hidden structure
        self.hidden_layer = nn.Sequential(
            nn.Linear(self.input_units, self.hidden_units),
            nn.ReLU(),
        )

        # Defining MLP output
        self.output_layer = nn.Sequential(
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.hidden_units, 1),
        )

    # Method for the forward pass
    def forward(self, x):
        x = self.output_layer(self.hidden_layer(x))
        return x

    # Method for obtaining prediction with predictive uncertainty
    def get_uncertainty(self, X, device, MC_size, normalize=None):

        # Sending everything to device
        self.to(device)
        X = torch.tensor(X).to(device)

        # Preparing predictions container
        n_samples = X.shape[0]
        preds = torch.empty(size=(X.shape[0], MC_size))

        torch.set_grad_enabled(False)   # Deactivating grad

        # Getting the predictions with Monte-Carlo Sampling
        for i in range(n_samples):
            new_X = X[i].repeat(MC_size, 1)
            preds[i] = (torch.squeeze(self(new_X)))

        # If we are using gpu re-convert the predictions
        if device == 'cuda':
            preds = preds.cpu()

        # If we want to give unnormalized predictions we re-mutiply std and add mean of the dataset
        if normalize is not None:
            preds = preds * normalize[1] + normalize[0]

        # Getting mean (prediction) and variance (uncertainty) of the prediction
        pred = np.array(torch.mean(preds.detach(), 1).numpy())
        unc = np.array(torch.var(preds.detach(), 1).numpy())

        torch.set_grad_enabled(True)    # Reactivating grad

        return pred, unc
