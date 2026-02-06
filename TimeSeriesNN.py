import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA
from sklearn.model_selection import TimeSeriesSplit
#------------------------------------------
#       MULTIVARIANT TIME-SERIES DATA
#------------------------------------------

np.random.seed(42)

timesteps = 1000
time = np.arange(timesteps)

data = pd.DataFrame({
    "temp": 10 + 10*np.sin(2*np.pi*time/365) + np.random.normal(0, 0.5, timesteps),
    "humidity": 50 + 20*np.sin(2*np.pi*time/180) + np.random.normal(0, 1, timesteps),
    "wind": 5 + np.random.normal(0, 0.3, timesteps),
    "pressure": 1013 + np.random.normal(0, 0.8, timesteps),
    "energy": 100 + 0.05*time + 5*np.sin(2*np.pi*time/30) + np.random.normal(0, 2, timesteps)
})

target_col = "energy"
features = data.columns.tolist()
#------------------------------------------
#                SCALING
#------------------------------------------

scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data)
#------------------------------------------
#             Sequence Dataset
#------------------------------------------


SEQ_LEN = 30
HORIZON = 5

class TimeSeriesDataset(Dataset):
    def __init__(self, data, seq_len, horizon):
        self.X, self.y = [], []
        for i in range(len(data) - seq_len - horizon):
            self.X.append(data[i:i+seq_len])
            self.y.append(data[i+seq_len:i+seq_len+horizon, -1])
        self.X = torch.tensor(self.X, dtype=torch.float32)
        self.y = torch.tensor(self.y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
#------------------------------------------
#          Train-Test split
#------------------------------------------

train_size = int(0.8 * len(scaled_data))
train_data = scaled_data[:train_size]
test_data = scaled_data[train_size:]

train_ds = TimeSeriesDataset(train_data, SEQ_LEN, HORIZON)
test_ds = TimeSeriesDataset(test_data, SEQ_LEN, HORIZON)

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)

#------------------------------------------
#                LSTM 
#------------------------------------------

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, horizon):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, horizon)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

# ------------------------------------------
#   TIME-SERIES CROSS-VALIDATION & TUNING
# ------------------------------------------

tscv = TimeSeriesSplit(n_splits=3)

param_grid = {
    "hidden_size": [32, 64],
    "num_layers": [1, 2],
    "lr": [0.001, 0.0005]
}

best_score = float("inf")
best_params = None

for hidden_size in param_grid["hidden_size"]:
    for num_layers in param_grid["num_layers"]:
        for lr in param_grid["lr"]:

            fold_scores = []

            for train_idx, val_idx in tscv.split(train_data):
                train_fold = train_data[train_idx]
                val_fold = train_data[val_idx]

                train_ds_fold = TimeSeriesDataset(train_fold, SEQ_LEN, HORIZON)
                val_ds_fold = TimeSeriesDataset(val_fold, SEQ_LEN, HORIZON)

                train_loader_fold = DataLoader(train_ds_fold, batch_size=32, shuffle=False)
                val_loader_fold = DataLoader(val_ds_fold, batch_size=32, shuffle=False)

                model_cv = LSTMModel(
                    input_size=len(features),
                    hidden_size=hidden_size,
                    num_layers=num_layers,
                    horizon=HORIZON
                ).to(device)

                optimizer_cv = torch.optim.Adam(model_cv.parameters(), lr=lr)
                criterion = nn.MSELoss()

                # Train briefly for CV
                for _ in range(5):
                    for X, y in train_loader_fold:
                        X, y = X.to(device), y.to(device)
                        optimizer_cv.zero_grad()
                        loss = criterion(model_cv(X), y)
                        loss.backward()
                        optimizer_cv.step()

                # Validation
                model_cv.eval()
                val_preds, val_actuals = [], []

                with torch.no_grad():
                    for X, y in val_loader_fold:
                        X = X.to(device)
                        val_preds.append(model_cv(X).cpu().numpy())
                        val_actuals.append(y.numpy())

                val_preds = np.vstack(val_preds)
                val_actuals = np.vstack(val_actuals)

                fold_scores.append(mean_absolute_error(val_actuals, val_preds))

            avg_score = np.mean(fold_scores)

            if avg_score < best_score:
                best_score = avg_score
                best_params = (hidden_size, num_layers, lr)

print("Best CV Params:", best_params)

#------------------------------------------
#             Train Model
#------------------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = LSTMModel(
    input_size=len(features),
    hidden_size=best_params[0],
    num_layers=best_params[1],
    horizon=HORIZON
).to(device)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=best_params[2])

EPOCHS = 25

for epoch in range(EPOCHS):
    model.train()
    losses = []
    for X, y in train_loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        preds = model(X)
        loss = criterion(preds, y)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {np.mean(losses):.4f}")

#------------------------------------------
#         Evaluation Metrics
#------------------------------------------

model.eval()
preds, actuals = [], []

with torch.no_grad():
    for X, y in test_loader:
        X = X.to(device)
        out = model(X)
        preds.append(out.cpu().numpy())
        actuals.append(y.numpy())

preds = np.vstack(preds)
actuals = np.vstack(actuals)

mae = mean_absolute_error(actuals, preds)
rmse = np.sqrt(mean_squared_error(actuals, preds))
mape = np.mean(np.abs((actuals - preds) / actuals)) * 100

print("LSTM Performance")
print("MAE:", mae)
print("RMSE:", rmse)
print("MAPE:", mape)

target_series = data[target_col]
#------------------------------------------
#         Baseline Models
#------------------------------------------
# ARIMA
arima_model = ARIMA(target_series[:train_size], order=(2,1,2)).fit()
arima_forecast = arima_model.forecast(steps=len(target_series)-train_size)

# Exponential Smoothing
ets_model = ExponentialSmoothing(
    target_series[:train_size],
    trend="add",
    seasonal="add",
    seasonal_periods=30
).fit()

ets_forecast = ets_model.forecast(len(target_series)-train_size)
#------------------------------------------
#           Explainability
#------------------------------------------


def integrated_gradients(model, input_tensor, baseline=None, steps=50):
    if baseline is None:
        baseline = torch.zeros_like(input_tensor)

    scaled_inputs = [
        baseline + (float(i)/steps)*(input_tensor-baseline)
        for i in range(steps+1)
    ]

    grads = []
    for inp in scaled_inputs:
        inp.requires_grad = True
        output = model(inp).sum()
        model.zero_grad()
        output.backward()
        grads.append(inp.grad.detach().clone())

    avg_grads = torch.mean(torch.stack(grads), dim=0)
    integrated_grad = (input_tensor - baseline) * avg_grads
    return integrated_grad

# ------------------------------------------
#   INTEGRATED GRADIENTS – MULTI-SAMPLE
# ------------------------------------------

ig_importances = []

for idx in range(10):  # analyze multiple samples
    sample_input, _ = test_ds[idx]
    sample_input = sample_input.unsqueeze(0).to(device)
    ig = integrated_gradients(model, sample_input)
    ig_importances.append(ig.abs().mean(dim=1).cpu().numpy()[0])

ig_importances = np.array(ig_importances)
mean_importance = ig_importances.mean(axis=0)

plt.figure(figsize=(10,5))
plt.bar(features, mean_importance)
plt.title("Average Feature Importance Across Time (Integrated Gradients)")
plt.xticks(rotation=45)
plt.show()

# ------------------------------------------
#   PRODUCTION-READY PREDICTION FUNCTION
# ------------------------------------------

def predict_energy(model, input_sequence):
    """
    input_sequence: numpy array of shape (SEQ_LEN, num_features)
    """
    model.eval()
    input_tensor = torch.tensor(input_sequence, dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        prediction = model(input_tensor).cpu().numpy()
    return prediction



