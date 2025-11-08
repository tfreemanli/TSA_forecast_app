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

def update_sarima(model, new_value, full_series=None):
    try:
        # 如果模型支持增量更新（statsmodels >= 0.12）
        if hasattr(model, "append"):
            updated_model = model.append([new_value], refit=False)
            print("✅ SARIMA was appended.")
        else:
            raise AttributeError("Current SARIMA does not support append()")
    except Exception as e:
        print(f"⚠️ Increamental update failed: {e}")
        if full_series is not None:
            print("🔁 Incremental update failed, training model with full series...")
            updated_model = train_sarima(full_series)
        else:
            raise RuntimeError("Full series is not provided for training.")

    # 保存更新后的模型
    with open(os.path.join(MODEL_DIR, "sarima_model.pkl"), "wb") as f:
        pickle.dump(updated_model, f)

    return updated_model