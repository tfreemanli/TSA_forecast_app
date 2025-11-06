import os
import pandas as pd
import matplotlib.pyplot as plt

OUTPUT_DIR = os.path.join("..", "output")

def plot_forecast(actual_df, forecast_df, date):
    plt.figure(figsize=(12,6))

    # 过去60天的实际销售
    plt.plot(actual_df["date"].iloc[-60:], actual_df["sales"].iloc[-60:],
             label="Actual", color="blue", linewidth=2)

    # 过去14天（如果存在历史预测记录，可从forecast_data.csv读取）
    past_forecast_path = os.path.join("..", "data", "forecast_data.csv")
    if os.path.exists(past_forecast_path):
        hist_forecast = pd.read_csv(past_forecast_path)
        last_14 = hist_forecast.tail(14)
        plt.plot(last_14["date"], last_14["sarima"], label="SARIMA Past Forecast",
                 color="green", linestyle="--", alpha=0.7)
        plt.plot(last_14["date"], last_14["adaptive"], label="Adaptive Past Forecast",
                 color="red", linestyle="--", alpha=0.7)

    # 当前预测（未来7天）
    plt.plot(forecast_df["date"], forecast_df["sarima"], label="SARIMA New Forecast",
             color="green", linewidth=2)
    plt.plot(forecast_df["date"], forecast_df["adaptive"], label="Adaptive New Forecast",
             color="red", linewidth=2)

    plt.xticks(rotation=45)
    plt.title("Fashion Store Daily Sales Forecast")
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