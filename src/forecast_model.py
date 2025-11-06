import os
import pickle
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

MODEL_DIR = os.path.join("..", "model")

def train_sarima(sales_series):
    """训练 SARIMA 模型"""
    model = SARIMAX(sales_series, order=(1,1,1), seasonal_order=(1,1,1,7))
    model_fit = model.fit(disp=False)
    with open(os.path.join(MODEL_DIR, "sarima_model.pkl"), "wb") as f:
        pickle.dump(model_fit, f)
    return model_fit

def load_sarima():
    with open(os.path.join(MODEL_DIR, "sarima_model.pkl"), "rb") as f:
        return pickle.load(f)

def forecast_sarima(model, steps=7):
    """预测未来若干天"""
    return model.forecast(steps=steps)
