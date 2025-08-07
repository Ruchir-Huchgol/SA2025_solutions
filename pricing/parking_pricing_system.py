# Advanced Dynamic Parking Pricing System
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime
from datetime import datetime, timedelta
import pathway as pw
import bokeh.plotting
from bokeh.models import ColumnDataSource, HoverTool
from bokeh.layouts import column, row
from bokeh.plotting import figure
import panel as pn
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Enhanced Schema for comprehensive feature set
class EnhancedParkingSchema(pw.Schema):
    Timestamp: str
    ParkingLotID: int
    Latitude: float
    Longitude: float
    Occupancy: int
    Capacity: int
    QueueLength: int
    VehicleType: str  # 'car', 'bike', 'truck'
    TrafficLevel: float
    IsSpecialDay: int
    BasePrice: float

# Advanced Neural Network Implementation (using only numpy)
class SimpleNeuralNetwork:
    """
    A simple feedforward neural network implemented from scratch
    for demand prediction in parking pricing
    """
    
    def __init__(self, input_size, hidden_size=10, output_size=1, learning_rate=0.01):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        
        # Initialize weights and biases
        self.W1 = np.random.randn(input_size, hidden_size) * 0.1
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * 0.1
        self.b2 = np.zeros((1, output_size))
        
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    
    def sigmoid_derivative(self, x):
        return x * (1 - x)
    
    def forward(self, X):
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self.sigmoid(self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = self.sigmoid(self.z2)
        return self.a2
    
    def backward(self, X, y, output):
        m = X.shape[0]
        
        # Calculate gradients
        dz2 = output - y
        dW2 = (1/m) * np.dot(self.a1.T, dz2)
        db2 = (1/m) * np.sum(dz2, axis=0, keepdims=True)
        
        dz1 = np.dot(dz2, self.W2.T) * self.sigmoid_derivative(self.a1)
        dW1 = (1/m) * np.dot(X.T, dz1)
        db1 = (1/m) * np.sum(dz1, axis=0, keepdims=True)
        
        # Update weights
        self.W2 -= self.learning_rate * dW2
        self.b2 -= self.learning_rate * db2
        self.W1 -= self.learning_rate * dW1
        self.b1 -= self.learning_rate * db1
    
    def train(self, X, y, epochs=1000):
        for epoch in range(epochs):
            output = self.forward(X)
            self.backward(X, y, output)
            
            if epoch % 100 == 0:
                loss = np.mean((output - y) ** 2)
                print(f"Epoch {epoch}, Loss: {loss:.6f}")
    
    def predict(self, X):
        return self.forward(X)

# LSTM-like implementation using only numpy
class SimpleLSTM:
    """
    Simplified LSTM implementation for time series prediction
    """
    
    def __init__(self, input_size, hidden_size=20, sequence_length=5):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.sequence_length = sequence_length
        
        # Initialize weights for LSTM gates
        self.Wf = np.random.randn(input_size + hidden_size, hidden_size) * 0.1
        self.Wi = np.random.randn(input_size + hidden_size, hidden_size) * 0.1
        self.Wc = np.random.randn(input_size + hidden_size, hidden_size) * 0.1
        self.Wo = np.random.randn(input_size + hidden_size, hidden_size) * 0.1
        
        self.bf = np.zeros((1, hidden_size))
        self.bi = np.zeros((1, hidden_size))
        self.bc = np.zeros((1, hidden_size))
        self.bo = np.zeros((1, hidden_size))
        
        # Output layer
        self.Wy = np.random.randn(hidden_size, 1) * 0.1
        self.by = np.zeros((1, 1))
        
        self.hidden_states = []
        self.cell_states = []
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    
    def tanh(self, x):
        return np.tanh(np.clip(x, -500, 500))
    
    def forward_step(self, x, h_prev, c_prev):
        # Concatenate input and previous hidden state
        concat = np.concatenate([x, h_prev], axis=1)
        
        # Forget gate
        f = self.sigmoid(np.dot(concat, self.Wf) + self.bf)
        
        # Input gate
        i = self.sigmoid(np.dot(concat, self.Wi) + self.bi)
        
        # Candidate values
        c_candidate = self.tanh(np.dot(concat, self.Wc) + self.bc)
        
        # Output gate
        o = self.sigmoid(np.dot(concat, self.Wo) + self.bo)
        
        # Update cell state
        c = f * c_prev + i * c_candidate
        
        # Update hidden state
        h = o * self.tanh(c)
        
        return h, c
    
    def predict(self, sequence):
        h = np.zeros((1, self.hidden_size))
        c = np.zeros((1, self.hidden_size))
        
        for i in range(len(sequence)):
            x = sequence[i].reshape(1, -1)
            h, c = self.forward_step(x, h, c)
        
        # Output prediction
        output = np.dot(h, self.Wy) + self.by
        return output[0, 0]

# Advanced Pricing Models
class AdvancedPricingEngine:
    """
    Comprehensive pricing engine with multiple models
    """
    
    def __init__(self, base_price=10):
        self.base_price = base_price
        self.scaler = StandardScaler()
        self.nn_model = None
        self.lstm_model = None
        self.price_history = {}
        self.demand_history = {}
        
        # Vehicle type weights
        self.vehicle_weights = {
            'car': 1.0,
            'bike': 0.5,
            'truck': 1.5
        }
        
        # Initialize models
        self.initialize_models()
    
    def initialize_models(self):
        """Initialize ML models"""
        # Neural network for demand prediction
        self.nn_model = SimpleNeuralNetwork(input_size=7, hidden_size=15, output_size=1)
        
        # LSTM for time series forecasting
        self.lstm_model = SimpleLSTM(input_size=3, hidden_size=20, sequence_length=5)
    
    def haversine_distance(self, lat1, lon1, lat2, lon2):
        """Calculate distance between two points using Haversine formula"""
        R = 6371  # Earth's radius in kilometers
        
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        
        return R * c
    
    def calculate_competitor_prices(self, current_lat, current_lon, all_lots_data, max_distance=2.0):
        """Calculate nearby competitor prices"""
        competitor_prices = []
        
        for lot_id, lot_data in all_lots_data.items():
            if lot_data['lat'] != current_lat or lot_data['lon'] != current_lon:
                distance = self.haversine_distance(
                    current_lat, current_lon,
                    lot_data['lat'], lot_data['lon']
                )
                
                if distance <= max_distance:
                    competitor_prices.append(lot_data.get('current_price', self.base_price))
        
        return competitor_prices
    
    def model_1_linear_pricing(self, previous_price, occupancy, capacity):
        """Model 1: Baseline Linear Model"""
        alpha = 0.1
        occupancy_rate = occupancy / capacity
        price_change = alpha * occupancy_rate
        
        new_price = previous_price + price_change
        return max(self.base_price * 0.5, min(new_price, self.base_price * 2))
    
    def model_2_demand_based_pricing(self, occupancy, capacity, queue_length, 
                                   traffic_level, is_special_day, vehicle_type):
        """Model 2: Advanced Demand-Based Pricing"""
        
        # Normalize inputs
        occupancy_rate = occupancy / capacity
        queue_normalized = min(queue_length / 10, 1.0)  # Normalize to [0,1]
        traffic_normalized = min(traffic_level / 100, 1.0)  # Assuming max traffic is 100
        vehicle_weight = self.vehicle_weights.get(vehicle_type, 1.0)
        
        # Enhanced demand function with non-linear components
        base_demand = occupancy_rate ** 1.5  # Non-linear occupancy effect
        queue_pressure = queue_normalized * 2.0  # High queue impact
        traffic_factor = traffic_normalized * 0.8  # Traffic congestion effect
        special_day_boost = is_special_day * 0.5  # Special day premium
        vehicle_adjustment = (vehicle_weight - 1.0) * 0.3  # Vehicle type adjustment
        
        # Time-based demand (assuming hour can be extracted from timestamp)
        time_factor = self.calculate_time_factor()
        
        # Calculate total demand
        total_demand = (base_demand + queue_pressure + traffic_factor + 
                       special_day_boost + vehicle_adjustment + time_factor)
        
        # Smooth demand normalization using sigmoid
        normalized_demand = 1 / (1 + np.exp(-2 * (total_demand - 1)))
        
        # Calculate price with smooth variation
        price_multiplier = 0.5 + 1.5 * normalized_demand  # Range: [0.5, 2.0]
        final_price = self.base_price * price_multiplier
        
        return final_price
    
    def model_3_competitive_pricing(self, base_price, competitor_prices, 
                                  occupancy, capacity, queue_length):
        """Model 3: Competitive Pricing with Market Intelligence"""
        
        if not competitor_prices:
            return base_price
        
        avg_competitor_price = np.mean(competitor_prices)
        min_competitor_price = np.min(competitor_prices)
        
        occupancy_rate = occupancy / capacity
        
        # Competitive adjustment logic
        if occupancy_rate > 0.9:  # High occupancy
            if base_price > avg_competitor_price:
                # Reduce price slightly to stay competitive
                competitive_price = base_price * 0.95
            else:
                # Increase price due to high demand
                competitive_price = base_price * 1.1
        
        elif occupancy_rate < 0.3:  # Low occupancy
            # Aggressive pricing to attract customers
            competitive_price = min(base_price * 0.8, min_competitor_price * 0.95)
        
        else:  # Normal occupancy
            # Price competitively around market average
            competitive_price = (base_price + avg_competitor_price) / 2
        
        # Add queue pressure
        if queue_length > 5:
            competitive_price *= 1.2
        
        return max(self.base_price * 0.5, min(competitive_price, self.base_price * 2))
    
    def calculate_time_factor(self):
        """Calculate time-based demand factor"""
        current_hour = datetime.now().hour
        
        # Peak hours: 8-10 AM and 5-7 PM
        if 8 <= current_hour <= 10 or 17 <= current_hour <= 19:
            return 0.3  # High demand periods
        elif 11 <= current_hour <= 16:
            return 0.1  # Normal business hours
        else:
            return -0.2  # Low demand periods
    
    def predict_with_neural_network(self, features):
        """Use neural network for demand prediction"""
        if self.nn_model is None:
            return 0.5  # Default prediction
        
        # Normalize features
        features_array = np.array(features).reshape(1, -1)
        prediction = self.nn_model.predict(features_array)
        
        return prediction[0, 0]
    
    def predict_with_lstm(self, time_series_data):
        """Use LSTM for time series forecasting"""
        if len(time_series_data) < 5:
            return 0.5  # Default prediction
        
        # Use last 5 data points for prediction
        sequence = np.array(time_series_data[-5:])
        prediction = self.lstm_model.predict(sequence)
        
        return prediction

# Enhanced Pathway Processing with Advanced Models
def create_enhanced_pricing_stream(data):
    """Create enhanced pricing stream with advanced models"""
    
    pricing_engine = AdvancedPricingEngine()
    
    # Parse timestamp and add time features
    fmt = "%Y-%m-%d %H:%M:%S"
    enhanced_data = data.with_columns(
        t=data.Timestamp.dt.strptime(fmt),
        hour=data.Timestamp.dt.strptime(fmt).dt.hour,
        day_of_week=data.Timestamp.dt.strptime(fmt).dt.dayofweek,
        occupancy_rate=data.Occupancy / data.Capacity,
        queue_pressure=data.QueueLength / (data.QueueLength + 1),  # Avoid division by zero
    )
    
    # Model 1: Linear pricing
    linear_pricing = enhanced_data.with_columns(
        price_linear=pricing_engine.base_price + 
                    0.1 * (enhanced_data.Occupancy / enhanced_data.Capacity)
    )
    
    # Model 2: Demand-based pricing
    demand_pricing = enhanced_data.with_columns(
        demand_score=(
            (enhanced_data.Occupancy / enhanced_data.Capacity) ** 1.5 +
            enhanced_data.QueueLength * 0.2 +
            enhanced_data.TrafficLevel * 0.01 +
            enhanced_data.IsSpecialDay * 0.5
        ),
        price_demand=pricing_engine.base_price * (
            0.5 + 1.5 / (1 + pw.apply(np.exp, -2 * (
                (enhanced_data.Occupancy / enhanced_data.Capacity) ** 1.5 +
                enhanced_data.QueueLength * 0.2 +
                enhanced_data.TrafficLevel * 0.01 +
                enhanced_data.IsSpecialDay * 0.5 - 1
            )))
        )
    )
    
    return demand_pricing

# Advanced Visualization Functions
def create_advanced_visualizations():
    """Create comprehensive visualization dashboard"""
    
    def multi_model_plotter(source):
        """Create multiple model comparison plot"""
        
        # Price comparison plot
        price_fig = figure(
            height=400, width=800,
            title="Dynamic Pricing Models Comparison",
            x_axis_type="datetime",
            tools="pan,wheel_zoom,box_zoom,reset,save"
        )
        
        # Add lines for different models
        price_fig.line("t", "price_linear", source=source, 
                      line_width=2, color="blue", legend_label="Linear Model")
        price_fig.line("t", "price_demand", source=source, 
                      line_width=2, color="red", legend_label="Demand Model")
        
        # Add hover tool
        hover = HoverTool(tooltips=[
            ("Time", "@t{%F %T}"),
            ("Linear Price", "@price_linear{$0.00}"),
            ("Demand Price", "@price_demand{$0.00}"),
            ("Occupancy", "@Occupancy"),
            ("Queue Length", "@QueueLength")
        ], formatters={"@t": "datetime"})
        
        price_fig.add_tools(hover)
        price_fig.legend.location = "top_left"
        
        return price_fig
    
    def occupancy_heatmap(source):
        """Create occupancy heatmap"""
        
        heatmap_fig = figure(
            height=300, width=800,
            title="Occupancy Rate Over Time",
            x_axis_type="datetime"
        )
        
        heatmap_fig.circle("t", "occupancy_rate", source=source,
                          size=10, alpha=0.7, color="green")
        
        return heatmap_fig
    
    def demand_metrics_plot(source):
        """Create demand metrics visualization"""
        
        metrics_fig = figure(
            height=300, width=800,
            title="Demand Metrics Dashboard",
            x_axis_type="datetime"
        )
        
        metrics_fig.line("t", "demand_score", source=source,
                        line_width=2, color="orange", legend_label="Demand Score")
        metrics_fig.line("t", "queue_pressure", source=source,
                        line_width=2, color="purple", legend_label="Queue Pressure")
        
        metrics_fig.legend.location = "top_left"
        
        return metrics_fig
    
    return multi_model_plotter, occupancy_heatmap, demand_metrics_plot

# Main execution pipeline
def run_advanced_pricing_system():
    """Run the complete advanced pricing system"""
    
    # Load and prepare data
    print("Loading and preparing data...")
    
    # Note: This would be replaced with actual data loading
    # For now, we'll use the provided sample structure
    
    # Create enhanced schema and processing pipeline
    pricing_stream = create_enhanced_pricing_stream(data)
    
    # Create visualizations
    multi_plotter, occupancy_plotter, demand_plotter = create_advanced_visualizations()
    
    # Set up real-time dashboard
    price_viz = pricing_stream.plot(multi_plotter, sorting_col="t")
    occupancy_viz = pricing_stream.plot(occupancy_plotter, sorting_col="t")
    demand_viz = pricing_stream.plot(demand_plotter, sorting_col="t")
    
    # Create comprehensive dashboard
    dashboard = pn.Column(
        "## Advanced Dynamic Parking Pricing System",
        price_viz,
        occupancy_viz,
        demand_viz
    )
    
    return dashboard

# Performance evaluation metrics
def calculate_pricing_performance_metrics(actual_occupancy, predicted_prices):
    """Calculate performance metrics for pricing strategy"""
    
    # Revenue optimization metric
    revenue_efficiency = np.mean(predicted_prices * actual_occupancy)
    
    # Price stability metric (lower is better)
    price_volatility = np.std(predicted_prices)
    
    # Utilization efficiency
    utilization_rate = np.mean(actual_occupancy)
    
    return {
        'revenue_efficiency': revenue_efficiency,
        'price_volatility': price_volatility,
        'utilization_rate': utilization_rate,
        'pricing_score': revenue_efficiency / (1 + price_volatility)
    }

# Example usage and testing
if __name__ == "__main__":
    print("Advanced Dynamic Parking Pricing System")
    print("=" * 50)
    
    # Initialize pricing engine
    engine = AdvancedPricingEngine()
    
    # Test models with sample data
    print("\nTesting pricing models:")
    
    # Model 1 test
    linear_price = engine.model_1_linear_pricing(10, 75, 100)
    print(f"Linear Model Price: ${linear_price:.2f}")
    
    # Model 2 test
    demand_price = engine.model_2_demand_based_pricing(75, 100, 8, 65, 1, 'car')
    print(f"Demand Model Price: ${demand_price:.2f}")
    
    # Model 3 test
    competitive_price = engine.model_3_competitive_pricing(
        demand_price, [12.5, 11.8, 13.2], 75, 100, 8
    )
    print(f"Competitive Model Price: ${competitive_price:.2f}")
    
    print("\nSystem ready for real-time deployment!")
