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


def _load_forecast_data():
    """加载预测数据"""
    path = os.path.join(DATA_DIR, "forecast_data.csv")
    if os.path.exists(path):
        df = pd.read_csv(path)
        # 确保日期格式一致
        df['date'] = pd.to_datetime(df['date'], format='%d-%m-%Y').dt.strftime('%d-%m-%Y')
        return df
    else:
        # 如果文件不存在，创建空的DataFrame
        return pd.DataFrame(columns=['date', 'sarima', 'adaptive'])


def update_forecast_data(date, sarima_pred, adaptive_pred):
    """更新预测数据"""
    path = os.path.join(DATA_DIR, "forecast_data.csv")

    # 确保日期格式为dd-MM-yyyy
    if isinstance(date, str):
        date_str = pd.to_datetime(date, dayfirst=True).strftime('%d-%m-%Y')
    else:
        date_str = date.strftime('%d-%m-%Y')

    # 加载现有数据
    if os.path.exists(path):
        df = pd.read_csv(path)
        df['date'] = pd.to_datetime(df['date'], format='%d-%m-%Y').dt.strftime('%d-%m-%Y')
    else:
        df = pd.DataFrame(columns=['date', 'sarima', 'adaptive'])

    # 创建新数据行
    new_data = pd.DataFrame([{
        'date': date,
        'sarima': round(float(sarima_pred[0]), 2),
        'adaptive': round(float(adaptive_pred[0]), 2)
    }])

    # 如果日期已存在，则更新该行；否则添加新行
    mask = df['date'] == date
    if mask.any():
        df.loc[mask, ['sarima', 'adaptive']] = new_data[['sarima', 'adaptive']].values
    else:
        df = pd.concat([df, new_data], ignore_index=True)

    # 保存回CSV
    df.to_csv(path, index=False)
    return df