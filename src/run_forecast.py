import os
import pandas as pd
from datetime import datetime, timedelta
from data_manager import load_sales_data, update_sales_data, load_forecast_data, update_forecast_data
from forecast_model import train_sarima, load_sarima, forecast_sarima, update_sarima
from adaptive_module import train_adaptive_model, load_adaptive_model, forecast_adaptive, update_adaptive_model
from visualize import plot_forecast
import pickle

def main():
    while True:

        # 加载数据
        sales_df = load_sales_data()
        last_date = datetime.strptime(sales_df["date"].iloc[-1], "%d-%m-%Y")
        today_date = last_date + timedelta(days=1)
        today = today_date.strftime("%d-%m-%Y")
        print(f"System detected Sales-Last-Date：{last_date}")

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
        #forecast_history = load_forecast_data()

        # 获取过去14天的实际销售数据
        # past_days = 14
        # historical_data = sales_df[["date", "sales"]].copy()
        # historical_data = historical_data.tail(past_days)

        #forecast_dates = pd.date_range(today - timedelta(days=21), today + timedelta(days=7),freq="D")
        forecast_dates = pd.date_range(today_date, periods=7, freq="D")
        forecast_df = pd.DataFrame({
            "date": forecast_dates.strftime("%d-%m-%Y"),
            "weekday": forecast_dates.strftime("%A"),
            "sarima": sarima_pred.round(2),
            "adaptive": adaptive_pred
        })

        # 绘图输出
        plot_forecast(sales_df, forecast_df, today)

        #print(f"System is predicting the Sales of：")
        print(f"\n======== {today} Forecast Result ======")
        today_pred_sarima = float(forecast_df.loc[forecast_df["date"] == today, "sarima"].values[0])
        today_pred_adapt = float(forecast_df.loc[forecast_df["date"] == today, "adaptive"].values[0].round(2))
        print(f"SARIMA   forecast: {today_pred_sarima}")
        print(f"ADAPTIVE forecast: {today_pred_adapt}")
        print(f"-------------- Save ? ------------------")

        # === 是否保存今日预测结果 ===
        choice_save = input("> Do you want to save today's forecasting? (y/n): ")

        if choice_save.lower() == "y":
            # today_pred_sarima = float(forecast_df.loc[forecast_df["date"] == today, "sarima"].values[0])
            # today_pred_adapt = float(forecast_df.loc[forecast_df["date"] == today, "adaptive"].values[0])
            weekday = today_date.strftime("%A")

            update_forecast_data(today, weekday, today_pred_sarima, today_pred_adapt)
            print(f"Successfully saved the forecasting of {today}.")


        print(f"\n======= {today} Actual Sale =========")
        # 用户输入
        choice = input(f"> Would you like to input the Sales Volumn of {today}? (y/n): ")

        if choice.lower() == "y":
            try:
                val = float(input("Please type in the sales volumn: $"))
                update_sales_data(today, val)
                print("Sale data was updated.")

                try:
                    if today_date.weekday() == 6 : #if it is Sunday
                        print("🔄 It's Sunday, Re-Trainning SARIMA Model now ...")
                        sarima = train_sarima(sales_df["sales"])
                    else:
                        # update SARIMA
                        print("🔄 Updating SARIMA Model now ...")
                        sarima = update_sarima(sarima, val, sales_df["sales"])

                except Exception as e:
                    print(f"Error updating SARIMA model: {e}")

                try:
                    if today_date.weekday() == 6 : #if it is Sunday
                        print("🔄 It's Sunday, Re-Trainning ADAPTIVE Model now ...")
                        adaptive = train_adaptive_model(sales_df["sales"])
                    else:
                        # update ML modle
                        print("🔄 Updating ADAPTIVE Model now ...")
                        adaptive = update_adaptive_model(adaptive, sales_df["sales"], val)
                except Exception as e:
                    print(f"Error updating ADAPTIVE model: {e}")

                # print("✅ Models updated and saved successfully.")
            except ValueError:
                print("Invalid input. Please enter a valid number.")


        # --- 询问是否继续整个流程 ---
        print("\n============ Again or Quit? ==============")
        choice_loop = input("Would you like to conduct the Forecasting again?(y/n): ")
        if choice_loop.lower() != "y":
            print("\n Good Bye, Thanks for using！")
            break

if __name__ == "__main__":
    main()