import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_percentage_error

def time_series_analysis():
    # Load dataset
    sales_data = pd.read_csv("C:/Users/sraja/Downloads/sales_data_sample.csv", encoding='latin1')
    sales_data['ORDERDATE'] = pd.to_datetime(sales_data['ORDERDATE'])
    sales_data.set_index('ORDERDATE', inplace=True)

    # Visualize sales trends
    plt.figure(figsize=(10, 6))
    plt.plot(sales_data['SALES'], label='Sales') 
    plt.title('Sales Trends Over Time')
    plt.xlabel('Date')
    plt.ylabel('Sales')
    plt.legend()
    plt.show()

    # Moving average
    sales_data['Sales_MA'] = sales_data['SALES'].rolling(window=12).mean() 
    plt.figure(figsize=(10, 6))
    plt.plot(sales_data['SALES'], label='Sales')
    plt.plot(sales_data['Sales_MA'], label='Moving Average', color='orange')
    plt.title('Sales with Moving Average')
    plt.legend()
    plt.show()

    # ARIMA modeling
    model = ARIMA(sales_data['SALES'], order=(2, 1, 2)) 
    model_fit = model.fit()

    # Forecast future sales
    forecast_steps = 12  # Adjust for desired forecast range
    forecast = model_fit.forecast(steps=forecast_steps)

    # Plot forecast
    plt.figure(figsize=(10, 6))
    plt.plot(sales_data['SALES'], label='Historical Sales')
    plt.plot(pd.date_range(sales_data.index[-1], periods=forecast_steps, freq='M'),
             forecast, label='Forecast', color='red')
    plt.title('Sales Forecasting')
    plt.legend()
    plt.show()
    

    # Evaluate model
    
    
    print(f'MAPE: {mean_absolute_percentage_error(sales_data['SALES'][-forecast_steps:], forecast)}')

# Execute the function
time_series_analysis()
