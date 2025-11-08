import os
import pandas as pd
from datetime import datetime

DATA_DIR = os.path.join("..", "data")

def get_last_sales_date():
    """获取销售数据中的最后日期"""
    df = load_sales_data()
    last_date = pd.to_datetime(df['date'].iloc[-1], format='%d-%m-%Y')
    return last_date

def load_sales_data():
    path = os.path.join(DATA_DIR, "sales_data.csv")
    return pd.read_csv(path)

def load_training_sales_data():
    path = os.path.join(DATA_DIR, "sales_data_4Training_31Jul2025.csv")
    return pd.read_csv(path)

def load_forecast_data():
    path = os.path.join(DATA_DIR, "forecast_data.csv")
    return pd.read_csv(path)

def update_sales_data(date_str, sales_value):
    """更新销售数据并保存"""
    path = os.path.join(DATA_DIR, "sales_data.csv")
    df = pd.read_csv(path)
    if date_str in df["date"].values:
        df.loc[df["date"] == date_str, "sales"] = sales_value
    else:
        new_row = pd.DataFrame({"date": [date_str], "sales": [sales_value]})
        df = pd.concat([df, new_row], ignore_index=True)
    df.to_csv(path, index=False)
    return df


def update_forecast_data(date_str, weekday, sarima_val, adaptive_val):
    """新增或更新每日预测结果"""
    path = os.path.join(DATA_DIR, "forecast_data.csv")
    if os.path.exists(path):
        df = pd.read_csv(path)
    else:
        df = pd.DataFrame(columns=["date", "weekday", "sarima", "adaptive"])

    if date_str in df["date"].values:
        df.loc[df["date"] == date_str, ["sarima", "adaptive"]] = [sarima_val, adaptive_val]
    else:
        new_row = pd.DataFrame({
            "date": [date_str],
            "weekday": [weekday],
            "sarima": [sarima_val],
            "adaptive": [adaptive_val]
        })
        df = pd.concat([df, new_row], ignore_index=True)

    df.to_csv(path, index=False)
    return df