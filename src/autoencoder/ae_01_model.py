import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score

class Autoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

def train_ae(model, data, epochs=20, lr=1e-3, batch_size=128, loss_fn="mse"):
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss() if loss_fn == "mse" else nn.L1Loss()

    dataset = torch.utils.data.TensorDataset(torch.tensor(data, dtype=torch.float32))
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for _ in range(epochs):
        for batch in loader:
            x = batch[0]
            recon = model(x)
            loss = loss_fn(recon, x)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

def get_reconstruction_errors(model, data):
    model.eval()
    with torch.no_grad():
        x = torch.tensor(data, dtype=torch.float32)
        recon = model(x)
        errors = torch.mean((recon - x) ** 2, dim=1).numpy()
    return errors

def evaluate_threshold(y_true, errors, threshold):
    preds = (errors > threshold).astype(int)
    return {
        "precision": precision_score(y_true, preds, zero_division=0),
        "recall": recall_score(y_true, preds, zero_division=0),
        "f1_score": f1_score(y_true, preds, zero_division=0)
    }
