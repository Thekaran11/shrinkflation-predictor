import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from prophet import Prophet
import warnings

# Suppress prophet warnings for a cleaner terminal
warnings.filterwarnings('ignore')

print("Loading dataset and preparing features...")

# 1-4. Load and Clean Data (The Bulletproof Method)
file_path = 'Inflation-data.xlsx'
df = pd.read_excel(file_path, sheet_name='fcpi_m')

def is_date_col(col):
    col_str = str(col)
    if 'M' in col_str and col_str[:4].isdigit(): return True
    if col_str.isdigit() and len(col_str) == 6: return True
    return False

date_cols = [col for col in df.columns if is_date_col(col)]
id_cols = [col for col in df.columns if col not in date_cols]

df_long = pd.melt(df, id_vars=id_cols, value_vars=date_cols, var_name='Date_Str', value_name='Food_Inflation')
df_long['Date_Str'] = df_long['Date_Str'].astype(str).str.replace('M', '')
df_long['Date'] = pd.to_datetime(df_long['Date_Str'], format='%Y%m', errors='coerce')
df_long['Food_Inflation'] = pd.to_numeric(df_long['Food_Inflation'], errors='coerce')
df_long = df_long.dropna(subset=['Date', 'Food_Inflation'])

country_col = next((col for col in id_cols if str(col).lower() in ['country', 'country name']), None)
target_country = 'United States' 
country_data = df_long[df_long[country_col] == target_country].copy()
country_data = country_data[country_data['Date'].dt.year >= 2000].sort_values('Date')

# 5. Feature Engineering: Historical Shrinkflation Triggers
country_data['Volatility'] = country_data['Food_Inflation'].rolling(window=12).std()
country_data['Momentum'] = country_data['Food_Inflation'] - country_data['Food_Inflation'].shift(6)
vol_threshold = country_data['Volatility'].mean() + country_data['Volatility'].std()
country_data['Shrink_Risk_Flag'] = (country_data['Volatility'] > vol_threshold) & (country_data['Momentum'] > 0)
risk_periods = country_data[country_data['Shrink_Risk_Flag'] == True]

# ---------------------------------------------------------
# 6. ENTER AI: META'S PROPHET FORECASTING
# ---------------------------------------------------------
print(f"\nTraining AI Model to forecast {target_country} inflation through 2025...")

# Prophet requires columns to be named 'ds' (datestamp) and 'y' (value)
df_prophet = country_data[['Date', 'Food_Inflation']].rename(columns={'Date': 'ds', 'Food_Inflation': 'y'})

# Initialize the model (multiplicative handles economic scaling better)
model = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False, seasonality_mode='multiplicative')
model.fit(df_prophet)

# Create a dataframe stretching 24 months into the future
future_dates = model.make_future_dataframe(periods=24, freq='MS')

# Predict the future!
forecast = model.predict(future_dates)

print("\n--- AI Forecast for the Next 6 Months ---")
recent_forecast = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(24).head(6)
recent_forecast.columns = ['Date', 'Predicted_Inflation', 'Lower_Bound', 'Upper_Bound']
print(recent_forecast.to_string(index=False))

# ---------------------------------------------------------
# 7. INTERACTIVE VISUALIZATION (PLOTLY)
# ---------------------------------------------------------
print("\nGenerating Interactive Plotly Forecast...")

from prophet.plot import plot_plotly

# Create the interactive web chart
fig = plot_plotly(model, forecast, xlabel='Date', ylabel='Food Inflation Index')

# Update the title and layout for a modern look
fig.update_layout(
    title=f'AI Supply Chain & Shrinkflation Forecast: {target_country}',
    title_font_size=20,
    hovermode='x unified',
    template='plotly_white'
)

# This will automatically open a new tab in your web browser!
fig.show()