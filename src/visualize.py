import os
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

OUTPUT_DIR = os.path.join("..", "output")

def plot_forecast(actual_df, forecast_df, date):
    plt.figure(figsize=(12,6))

    actual_df = actual_df.copy()
    forecast_df = forecast_df.copy()

    actual_df["date"] = pd.to_datetime(actual_df["date"], format="%d-%m-%Y")
    forecast_df["date"] = pd.to_datetime(forecast_df["date"], format="%d-%m-%Y")

    # 过去14天的实际销售
    past_days = -15 # should be longer than the forecast_history below, to make sure alginment.
    plt.plot(actual_df["date"].iloc[past_days:], actual_df["sales"].iloc[past_days:],
             label="Actual", color="blue", linewidth=2)

    # 过去14天（如果存在历史预测记录，可从forecast_data.csv读取）
    past_forecast_path = os.path.join("..", "data", "forecast_data.csv")
    if os.path.exists(past_forecast_path):
        hist_forecast = pd.read_csv(past_forecast_path)
        last_14 = hist_forecast.tail(14).copy()
        last_14["date"] = pd.to_datetime(last_14["date"], format="%d-%m-%Y") - pd.Timedelta(days=1)
    else:
        last_14 = None

    # ---------- 计算 MAPE / AL ----------
    # 合并 forecast 与实际
    merged = pd.merge(last_14, actual_df, on="date", how="inner")
    merged = merged[merged["sales"] != 0]  # 避免除0
    if len(merged) > 0:
        mape_sarima = (abs(merged["sales"] - merged["sarima"]) / merged["sales"]).mean() * 100
        mape_adapt = (abs(merged["sales"] - merged["adaptive"]) / merged["sales"]).mean() * 100

        al_sarima = (abs(merged["sarima"] - merged["sales"]).sum() / merged["sales"].sum()) * len(merged)
        al_adapt = (abs(merged["adaptive"] - merged["sales"]).sum() / merged["sales"].sum()) * len(merged)
        print(f"len(merged):{len(merged)}")
    else:
        mape_sarima = mape_adapt = al_sarima = al_adapt = None

    if last_14 is not None:
        plt.plot(last_14["date"], last_14["sarima"], label="SARIMA Past Forecast",
                 color="green", linestyle="--", alpha=0.7)
        plt.plot(last_14["date"], last_14["adaptive"], label="Adaptive Past Forecast",
                 color="red", linestyle="--", alpha=0.7)

    # 当前预测（未来7天）
    forecast_df["date"] = pd.to_datetime(forecast_df["date"], format="%d-%m-%Y") - pd.Timedelta(days=1)
    plt.plot(forecast_df["date"], forecast_df["sarima"], label="SARIMA New Forecast",
             color="green", linewidth=2, marker="o", markersize=5)
    plt.plot(forecast_df["date"], forecast_df["adaptive"], label="Adaptive New Forecast",
             color="red", linewidth=2, marker="o", markersize=5)


    title = f"Sales Forecast vs Actual ({date} → +{len(forecast_df)} days)"
    if mape_sarima is not None:
        title += f"\nMAPE SARIMA: {mape_sarima:.2f}%, AL: {al_sarima:.2f} | " \
                 f"MAPE ADAPTIVE: {mape_adapt:.2f}%, AL: {al_adapt:.2f}"


    plt.xticks(rotation=45)
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Sales ($)")
    plt.legend()
    plt.tight_layout()
    #plt.savefig(os.path.join(OUTPUT_DIR, "forecast_plot.png"))

    # 确保输出目录存在
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 保存图表
    plt.savefig(os.path.join(OUTPUT_DIR, f"sales_forecast_{date}.png"),
                dpi=300,
                bbox_inches='tight')
    plt.show()
    #plt.close()

def plot_sarima_org_forecast(startdate, forecast_len, actual_df, forecast_df, mape=None, al=None):
    # 统一日期格式
    actual_df = actual_df.copy()
    forecast_df = forecast_df.copy()
    actual_df["date"] = pd.to_datetime(actual_df["date"], format="%d-%m-%Y")
    forecast_df["date"] = pd.to_datetime(forecast_df["date"], format="%d-%m-%Y")
    today_date = datetime.strptime(startdate, "%d-%m-%Y")
    end_date = today_date + timedelta(days=forecast_len - 1)

    # 只保留从 today - 3 天起的数据用于显示
    start_date = today_date - pd.Timedelta(days=3)
    mask = (actual_df["date"] >= start_date) & (actual_df["date"] <= end_date)
    actual_plot = actual_df.loc[mask]

    # ---------- 绘图 ----------
    plt.figure(figsize=(10, 5))
    plt.plot(actual_plot["date"], actual_plot["sales"],
             color="blue", linestyle="-", linewidth=2, label="Actual Sales")
    plt.plot(forecast_df["date"], forecast_df["sarima_org"],
             color="green", linestyle="--", linewidth=2, label="SARIMA Forecast")

    # ---------- 美化 ----------
    title = f"SARIMA Forecast vs Actual Sales ({startdate} → +{forecast_len} days)"
    if mape is not None and al is not None:
        title += f"\nMAPE = {mape:.2f}%   |   AL = {al:.2f} days"

    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Sales")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()

    # ---------- 显示 ----------
    plt.show()
    return