# Capstone 2025: Dynamic Pricing for Urban Parking Lots

# ✅ Section 1: Setup
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from geopy.distance import geodesic
from bokeh.plotting import figure, show, output_notebook
from bokeh.models import ColumnDataSource
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
import warnings
warnings.filterwarnings("ignore")
output_notebook()

# ✅ Section 2: Load Data
url = "YOUR_DATASET_URL_HERE"
df = pd.read_csv("dataset.csv")

# ✅ Section 3: Preprocessing
df['datetime'] = pd.to_datetime(df['timestamp'])
df.sort_values(by='datetime', inplace=True)

# Normalize continuous features
scaler = MinMaxScaler()
df[['occupancy', 'queue', 'traffic']] = scaler.fit_transform(df[['occupancy', 'queue', 'traffic']])

# Encode categorical features
df['vehicle_type_encoded'] = df['vehicle_type'].map({'car': 1.0, 'bike': 0.5, 'truck': 1.5})
df['is_special_day'] = df['day_type'].apply(lambda x: 1 if x == 'holiday' else 0)

# ✅ Section 4: Pricing Models

def baseline_pricing(occupancy, capacity, base_price=10, alpha=0.5):
    return base_price + alpha * (occupancy / capacity)

def demand_based_pricing(row, base_price=10, λ=1.0):
    demand_score = (0.4 * row['occupancy'] +
                    0.2 * row['queue'] +
                    0.15 * row['traffic'] +
                    0.15 * row['is_special_day'] +
                    0.1 * row['vehicle_type_encoded'])
    normalized = (demand_score - 0) / 1.5
    price = base_price * (1 + λ * normalized)
    return max(5, min(20, price))

def competitive_pricing(row, all_lots_df):
    current_location = (row['lat'], row['lon'])
    competitors = all_lots_df[all_lots_df['lot_id'] != row['lot_id']].copy()
    competitors['distance'] = competitors.apply(
        lambda x: geodesic(current_location, (x['lat'], x['lon'])).meters, axis=1)
    nearby = competitors[competitors['distance'] < 1000]
    avg_price = nearby['current_price'].mean() if not nearby.empty else row['current_price']
    adjustment = 0.2 * (avg_price - row['current_price'])
    return max(5, min(20, row['current_price'] + adjustment))

# ✅ Section 5: Time Series Forecasting - ARIMA

def arima_forecast(series, steps=3):
    model = ARIMA(series, order=(2, 1, 2))
    model_fit = model.fit()
    return model_fit.forecast(steps=steps)

# ✅ Section 6: Optional LSTM Setup (placeholder)
# Full LSTM pipeline can be added if required, here’s a stub

def placeholder_lstm_pipeline():
    print("Implement LSTM for demand forecasting if needed.")

# ✅ Section 7: Apply Models to Data

results = []
lot_ids = df['lot_id'].unique()

for lot_id in lot_ids:
    lot_df = df[df['lot_id'] == lot_id].copy()
    for i, row in lot_df.iterrows():
        base_price = 10
        price = demand_based_pricing(row, base_price)
        row['current_price'] = price
        row['final_price'] = competitive_pricing(row, lot_df)
        results.append(row)

result_df = pd.DataFrame(results)

# ✅ Section 8: Visualization with Bokeh

def plot_price_trends(df, lot_id):
    plot_df = df[df['lot_id'] == lot_id]
    p = figure(x_axis_type="datetime", title=f"Dynamic Price Trend - Lot {lot_id}", width=800)
    source = ColumnDataSource(plot_df)
    p.line(x='datetime', y='final_price', source=source, line_width=2, color='navy')
    show(p)

plot_price_trends(result_df, lot_id=1)

# ✅ Section 9: ARIMA Forecast for Lot 1 Occupancy
lot1_occ = df[df['lot_id'] == 1].set_index('datetime')['occupancy']
forecast = arima_forecast(lot1_occ, steps=5)
print("ARIMA Forecast (Next 5 Steps):")
print(forecast)

# ✅ Section 10: Export Final Results
result_df.to_csv("dynamic_pricing_output.csv", index=False)

# 📌 END
