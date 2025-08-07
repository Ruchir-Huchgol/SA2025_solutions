import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from geopy.distance import geodesic
from bokeh.plotting import figure, show, output_notebook
from bokeh.models import ColumnDataSource
import warnings
warnings.filterwarnings("ignore")

output_notebook()

# Load dataset
df = pd.read_csv('dataset.csv')

# Combine date and time columns into datetime
df['datetime'] = pd.to_datetime(df['LastUpdatedDate'] + ' ' + df['LastUpdatedTime'])

# Rename columns to internal standard names
df.rename(columns={
    'Longitude': 'lon',
    'Occupancy': 'occupancy',
    'VehicleType': 'vehicle_type',
    'TrafficConditionNearby': 'traffic',
    'QueueLength': 'queue',
    'IsSpecialDay': 'is_special_day'
}, inplace=True)

# Map vehicle types to weights
vehicle_type_weights = {'car': 1.0, 'bike': 0.5, 'truck': 1.5}
df['vehicle_type_encoded'] = df['vehicle_type'].map(vehicle_type_weights)

# Ensure is_special_day is integer (0 or 1)
df['is_special_day'] = df['is_special_day'].astype(int)

# Sort by datetime for time series consistency
df.sort_values(by='datetime', inplace=True)

# Normalize numeric columns for modeling
scaler = MinMaxScaler()
df[['occupancy', 'queue', 'traffic']] = scaler.fit_transform(df[['occupancy', 'queue', 'traffic']])

# Pricing Models

def baseline_pricing(occupancy, base_price=10, alpha=0.5):
    return base_price + alpha * occupancy  # occupancy is normalized 0-1

def demand_based_pricing(row, base_price=10, λ=1.0):
    demand_score = (
        0.4 * row['occupancy'] +
        0.2 * row['queue'] +
        0.15 * row['traffic'] +
        0.15 * row['is_special_day'] +
        0.1 * row['vehicle_type_encoded']
    )
    normalized = demand_score / 1.5  # max theoretical demand_score is ~1.5
    price = base_price * (1 + λ * normalized)
    return max(5, min(20, price))

def competitive_pricing(row, df_all):
    current_loc = (0, row['lon'])  # Note: latitude missing? If you have latitude add here
    competitors = df_all[df_all['lon'] != row['lon']].copy()  # simplistic competitor filter by lon diff
    # If you have latitude, replace with geodesic distance calculation:
    # competitors['distance'] = competitors.apply(lambda x: geodesic((lat, lon), (x['lat'], x['lon'])).meters, axis=1)
    
    # For now, skip geo distance for simplicity
    avg_comp_price = competitors['current_price'].mean() if not competitors.empty else row.get('current_price', 10)
    
    adjustment = 0.2 * (avg_comp_price - row.get('current_price', 10))
    final_price = row.get('current_price', 10) + adjustment
    return max(5, min(20, final_price))

# Example: Apply demand based pricing
df['current_price'] = df.apply(lambda row: demand_based_pricing(row), axis=1)

# Apply competitive pricing adjustment (using current_price from above)
df['final_price'] = df.apply(lambda row: competitive_pricing(row, df), axis=1)

# Visualization example using Bokeh
output_notebook()

def plot_pricing(df):
    p = figure(x_axis_type="datetime", title="Dynamic Pricing Over Time", width=900)
    source = ColumnDataSource(df)
    p.line('datetime', 'final_price', source=source, color='navy', legend_label='Price', line_width=2)
    p.line('datetime', 'occupancy', source=source, color='orange', legend_label='Occupancy (scaled)', line_width=2)
    p.legend.location = 'top_left'
    p.xaxis.axis_label = 'Time'
    p.yaxis.axis_label = 'Price / Occupancy'
    show(p)

plot_pricing(df)
