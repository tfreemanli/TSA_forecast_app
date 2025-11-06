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

def update_adaptive_model(model, series, new_value):

    if len(series) < 7:
        print("! Error: series is not enough for the increamantal update. 7 days is minimum.")
        return model

    # 准备输入特征
    X_new = np.array(series[-7:]).reshape(1, -1)
    y_new = np.array([new_value])

    # 从Pipeline中取出真正的回归器对象
    reg = model.named_steps["sgdregressor"]
    scaler = model.named_steps["standardscaler"]
    X_scaled = scaler.transform(X_new)

    # 进行增量更新
    reg.partial_fit(X_scaled, y_new)

    # 保存更新后的模型
    with open(os.path.join(MODEL_DIR, "adaptive_model.pkl"), "wb") as f:
        pickle.dump(model, f)

    print("✅ Adaptive Model updated successfully.")
    return model