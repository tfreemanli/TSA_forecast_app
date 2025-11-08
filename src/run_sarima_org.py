import os
import pandas as pd
from datetime import datetime, timedelta
from data_manager import load_sales_data, update_sales_data, load_training_sales_data, update_forecast_data
from sarima_model import *
from visualize import plot_sarima_org_forecast
import pickle

def run_sarima():

        # 加载数据
        train_df = load_training_sales_data()
        actual_df = load_sales_data()
        last_date = datetime.strptime(train_df["date"].iloc[-1], "%d-%m-%Y")
        today_date = last_date + timedelta(days=1)
        today = today_date.strftime("%d-%m-%Y")
        print(f"System detected Train-Last-Date：{last_date}")

        # 加载或训练模型
        try:
            sarima = load_sarima_org()
        except:
            sarima = train_sarima_org(train_df["sales"])

        # 预测未来28天
        forecast_len = 28
        sarima_pred = forecast_sarima_org(sarima, steps=forecast_len)

        forecast_dates = pd.date_range(today_date, periods=forecast_len, freq="D")
        forecast_df = pd.DataFrame({
            "date": forecast_dates.strftime("%d-%m-%Y"),
            "sarima_org": sarima_pred.round(2),
        })

        # ---------- 计算指标 ----------
        # 转为日期类型以便匹配
        actual_df["date"] = pd.to_datetime(actual_df["date"], format="%d-%m-%Y")
        forecast_df["date"] = pd.to_datetime(forecast_df["date"], format="%d-%m-%Y")

        # 取出重叠日期区间的数据
        merged = pd.merge(forecast_df, actual_df, on="date", how="inner")
        print(merged)

        if len(merged) > 0:
            # MAPE
            merged = merged[merged["sales"] != 0]  # 避免除0
            mape = (abs(merged["sales"] - merged["sarima_org"]) / merged["sales"]).mean() * 100

            # Adaptation Latency (AL)
            al = (abs(merged["sarima_org"] - merged["sales"]).sum() / merged["sales"].sum()) * forecast_len
        else:
            mape, al = None, None

        # 绘图输出
        plot_sarima_org_forecast(today, forecast_len, actual_df, forecast_df, mape, al)

if __name__ == "__main__":
    run_sarima()