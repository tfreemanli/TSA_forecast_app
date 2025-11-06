import os
import pickle
import numpy as np
import pandas as pd
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

MODEL_DIR = os.path.join("..", "model")

def create_lag_features(series, lag=7):
    df = pd.DataFrame({"y": series})
    for i in range(1, lag+1):
        df[f"lag_{i}"] = df["y"].shift(i)
    return df.dropna()

def train_adaptive_model(series):
    lag_df = create_lag_features(series)
    X, y = lag_df.drop("y", axis=1), lag_df["y"]
    model = make_pipeline(StandardScaler(), SGDRegressor(max_iter=1000, tol=1e-3))
    model.fit(X, y)
    with open(os.path.join(MODEL_DIR, "adaptive_model.pkl"), "wb") as f:
        pickle.dump(model, f)
    return model

def load_adaptive_model():
    with open(os.path.join(MODEL_DIR, "adaptive_model.pkl"), "rb") as f:
        return pickle.load(f)

def forecast_adaptive(model, recent_sales, steps=7):
    preds = []
    recent = list(recent_sales[-7:])
    for _ in range(steps):
        x = np.array(recent[-7:]).reshape(1, -1)
        pred = model.predict(x)[0]
        preds.append(pred)
        recent.append(pred)
    return preds
