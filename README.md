# Final-CJR-Project
Advanced Time Series Forecasting Using LSTM

Project Overview
This project focuses on time series forecasting using a multi-step Long Short-Term Memory (LSTM) neural network implemented in PyTorch. The project emphasizes model explainability, systematic hyperparameter tuning, and baseline comparison to ensure robust and interpretable forecasts.

Objectives
- Build a multi-step LSTM model for time series forecasting
- Compare performance with ARIMA and Exponential Smoothing baselines
- Apply Integrated Gradients for explainability
- Perform systematic hyperparameter tuning using time-series cross-validation
- Discuss production deployment considerations

Dataset Description
The dataset is a synthetic multivariate time series representing environmental variables and energy consumption. Preprocessing includes Min-Max normalization and sliding window sequence generation. A time-based split was used for training and testing.

Model Architecture
The model consists of stacked LSTM layers followed by a fully connected layer for multi-step forecasting. Mean Squared Error is used as the loss function with the Adam optimizer.

Hyperparameter Tuning
Key hyperparameters such as sequence length, hidden units, number of layers, and learning rate were tuned using time-series cross-validation.

Baseline Models
The LSTM model is compared with ARIMA and Exponential Smoothing models using MAE, RMSE, and MAPE metrics.

Model Explainability
Integrated Gradients is applied to analyze feature importance over time. The results indicate that recent timesteps contribute more strongly to predictions.

Production Deployment Considerations
The model can be deployed as a batch or real-time forecasting service. Monitoring for data drift and periodic retraining is recommended.

Tech Stack
Python, PyTorch, NumPy, Pandas, Scikit-learn, Statsmodels

Author
Sruthi Rajesh
