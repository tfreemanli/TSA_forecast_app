import os
import pandas as pd
from datetime import datetime, timedelta
from data_manager import load_sales_data, get_last_sales_date, update_sales_data, load_forecast_data, update_forecast_data, get_last_sales_date
from forecast_model import train_sarima, load_sarima, forecast_sarima
from adaptive_module import train_adaptive_model, load_adaptive_model, forecast_adaptive
from visualize import plot_forecast



# 加载数据
sales_df = load_sales_data()
last_date = datetime.strptime(sales_df["date"].iloc[-1], "%d-%m-%Y")
today_date = last_date + timedelta(days=1)
today = today_date.strftime("%d-%m-%Y")
print(f"System detected Sales-Last-Date：{today}")

# 加载或训练模型
try:
    sarima = load_sarima()
except:
    sarima = train_sarima(sales_df["sales"])

try:
    adaptive = load_adaptive_model()
except:
    adaptive = train_adaptive_model(sales_df["sales"])

# 预测未来7天
sarima_pred = forecast_sarima(sarima, steps=7)
adaptive_pred = forecast_adaptive(adaptive, sales_df["sales"].values, steps=7)

# 更新预测数据
#update_forecast_data(today, sarima_pred, adaptive_pred)

# 加载历史预测数据
forecast_history = load_forecast_data()

# 获取过去60天的实际销售数据
past_days = 60
historical_data = sales_df[["date", "sales"]].copy()
historical_data = historical_data.tail(past_days)

#forecast_dates = pd.date_range(today - timedelta(days=21), today + timedelta(days=7),freq="D")
forecast_dates = pd.date_range(today_date, periods=7, freq="D")
forecast_df = pd.DataFrame({
    "date": forecast_dates.strftime("%d-%m-%Y"),
    "weekday": forecast_dates.strftime("%A"),
    "sarima": sarima_pred.round(2),
    "adaptive": adaptive_pred
})

# # 创建完整的预测数据（历史+未来）
# forecast_dates = pd.date_range(today - timedelta(days=21), today + timedelta(days=7), freq="D")
# forecast_dates_str = [d.strftime('%d-%m-%Y') for d in forecast_dates]
#
# # 创建预测DataFrame
# forecast_df = pd.DataFrame({
#     'date': forecast_dates_str
# })
#
# # 合并历史预测数据
# forecast_history = forecast_history[forecast_history['date'].isin(forecast_dates_str)]
# forecast_df = pd.merge(forecast_df, forecast_history, on='date', how='left')
#
# # 添加新的预测数据
# today_str = today.strftime('%d-%m-%Y')
# for i in range(7):
#     pred_date = (today + timedelta(days=i)).strftime('%d-%m-%Y')
#     if pred_date not in forecast_df['date'].values:
#         new_row = pd.DataFrame([{
#             'date': pred_date,
#             'sarima': round(float(sarima_pred[i]), 2) if i < len(sarima_pred) else None,
#             'adaptive': round(float(adaptive_pred[i]), 2) if i < len(adaptive_pred) else None
#         }])
#         forecast_df = pd.concat([forecast_df, new_row], ignore_index=True)
#
# # 按日期排序
# forecast_df = forecast_df.sort_values('date').reset_index(drop=True)

# 绘图输出
plot_forecast(sales_df, forecast_df, today)

# 用户输入
choice = input(f"Would you like to input the Sales Volumn of {today}? (y/n): ")
if choice.lower() == "y":
    val = float(input("Please type in the sales volumn: $"))
    update_sales_data(today, val)
    print("Sale data was updated.")

print("Forecasting is done, please see output/sales_forecast.png")
