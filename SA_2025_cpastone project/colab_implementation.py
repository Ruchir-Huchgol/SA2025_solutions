# Complete Dynamic Parking Pricing System
# Summer Analytics 2025 - Enhanced Implementation

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# For systems without pathway, we'll create a simulation framework
try:
    import pathway as pw
    PATHWAY_AVAILABLE = True
except ImportError:
    PATHWAY_AVAILABLE = False
    print("Pathway not available. Using simulation framework instead.")

# =============================================================================
# ENHANCED DATA SCHEMA AND PREPROCESSING
# =============================================================================

def preprocess_parking_data(df):
    """
    Preprocess the parking data with actual column names
    """
    # Map actual columns to expected names
    column_mapping = {
        'LastUpdatedDate': 'Date',
        'LastUpdatedTime': 'Time',
        'TrafficConditionNearby': 'TrafficLevel'
    }
    
    df = df.rename(columns=column_mapping)
    
    
    df['Timestamp'] = pd.to_datetime(df['Date'] + ' ' + df['Time'],
        format='%d-%m-%Y %H:%M:%S', errors='coerce')
    
   
    df['Hour'] = df['Timestamp'].dt.hour
    df['DayOfWeek'] = df['Timestamp'].dt.dayofweek
    df['Month'] = df['Timestamp'].dt.month
    df['IsWeekend'] = df['DayOfWeek'].isin([5, 6]).astype(int)
    
   
    df['OccupancyRate'] = df['Occupancy'] / df['Capacity']
    df['AvailableSpaces'] = df['Capacity'] - df['Occupancy']
    df['QueuePressure'] = df['QueueLength'] / (df['QueueLength'] + 1)
    
    df = df.fillna(method='forward').fillna(method='backward')
    
    
    if df['TrafficLevel'].dtype == 'object':
        traffic_mapping = {'Low': 30, 'Medium': 60, 'High': 90}
        df['TrafficLevel'] = df['TrafficLevel'].map(traffic_mapping).fillna(60)
    
    
    if 'ParkingLotID' not in df.columns:
        df['ParkingLotID'] = 1
    
    return df.sort_values('Timestamp').reset_index(drop=True)

# =============================================================================
# ADVANCED PRICING ENGINE
# =============================================================================

class AdvancedPricingEngine:
    """
    Production-ready pricing engine with multiple sophisticated models
    """
    
    def __init__(self, base_price=10):
        self.base_price = base_price
        self.price_history = {}
        self.demand_history = {}
        self.model_weights = {
            'occupancy': 0.4,
            'queue': 0.3,
            'traffic': 0.2,
            'special_day': 0.1
        }
        
        # Vehicle type pricing multipliers
        self.vehicle_multipliers = {
            'Car': 1.0,
            'Bike': 0.5,
            'Truck': 1.5,
            'car': 1.0,
            'bike': 0.5,
            'truck': 1.5,
            'default': 1.0
        }
        
        # Time-based demand patterns
        self.time_patterns = {
            'peak_morning': (8, 10),
            'peak_evening': (17, 19),
            'business_hours': (10, 17),
            'off_peak': (19, 8)
        }
    
    def haversine_distance(self, lat1, lon1, lat2, lon2):
        """Calculate geographical distance between two points"""
        R = 6371  # Earth's radius in kilometers
        
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        
        return R * c
    
    def get_time_factor(self, hour):
        """Calculate time-based demand factor"""
        if self.time_patterns['peak_morning'][0] <= hour <= self.time_patterns['peak_morning'][1]:
            return 1.3  # High demand
        elif self.time_patterns['peak_evening'][0] <= hour <= self.time_patterns['peak_evening'][1]:
            return 1.4  # Highest demand
        elif self.time_patterns['business_hours'][0] <= hour <= self.time_patterns['business_hours'][1]:
            return 1.1  # Moderate demand
        else:
            return 0.8  # Low demand
    
    def calculate_occupancy_pressure(self, occupancy, capacity):
        """Calculate non-linear occupancy pressure"""
        if capacity == 0:
            return 0
        
        occupancy_rate = occupancy / capacity
        
        # Exponential pressure at high occupancy
        if occupancy_rate > 0.8:
            return occupancy_rate ** 2
        elif occupancy_rate > 0.6:
            return occupancy_rate ** 1.5
        else:
            return occupancy_rate
    
    def model_1_linear_pricing(self, current_price, occupancy, capacity):
        """
        Model 1: Enhanced Linear Pricing with Smoothing
        """
        if capacity == 0:
            return current_price
            
        occupancy_rate = occupancy / capacity
        
        # Linear adjustment with bounds
        alpha = 0.15  # Increased responsiveness
        price_adjustment = alpha * (occupancy_rate - 0.5)
        
        new_price = current_price + price_adjustment
        
        # Apply bounds and smoothing
        min_price = self.base_price * 0.5
        max_price = self.base_price * 2.0
        
        return np.clip(new_price, min_price, max_price)
    
    def model_2_demand_based_pricing(self, occupancy, capacity, queue_length,traffic_level, is_special_day, vehicle_type, hour):
        """
        Model 2: Advanced Demand-Based Pricing with Multiple Factors
        """
        
        # Normalized inputs
        occupancy_pressure = self.calculate_occupancy_pressure(occupancy, capacity)
        queue_pressure = min(queue_length / 10, 1.0)
        traffic_pressure = min(traffic_level / 100, 1.0)
        
        # Vehicle type multiplier
        vehicle_multiplier = self.vehicle_multipliers.get(vehicle_type, 1.0)
        
        # Time-based factor
        time_factor = self.get_time_factor(hour)
        
        # Special day boost
        special_day_factor = 1.2 if is_special_day else 1.0
        
        # Calculate weighted demand score
        demand_score = (
            occupancy_pressure * self.model_weights['occupancy'] +
            queue_pressure * self.model_weights['queue'] +
            traffic_pressure * self.model_weights['traffic'] +
            (special_day_factor - 1) * self.model_weights['special_day']
        )
        
        # Apply time factor
        demand_score *= time_factor
        
        # Sigmoid normalization for smooth transitions
        normalized_demand = 1 / (1 + np.exp(-3 * (demand_score - 0.5)))
        
        # Calculate final price
        price_multiplier = 0.5 + 1.5 * normalized_demand
        base_price_adjusted = self.base_price * vehicle_multiplier
        final_price = base_price_adjusted * price_multiplier
        
        return final_price
    
    def model_3_competitive_pricing(self, base_price, competitor_prices, 
                                  occupancy, capacity, queue_length, market_position='neutral'):
        """
        Model 3: Competitive Pricing with Market Intelligence
        """
        
        if not competitor_prices or capacity == 0:
            return base_price
        
        avg_competitor_price = np.mean(competitor_prices)
        min_competitor_price = np.min(competitor_prices)
        max_competitor_price = np.max(competitor_prices)
        
        occupancy_rate = occupancy / capacity
        queue_pressure = min(queue_length / 10, 1.0)
        
        # Market positioning strategy
        if market_position == 'premium':
            target_position = 1.1
        elif market_position == 'value':
            target_position = 0.9
        else:
            target_position = 1.0
        
        # Dynamic pricing logic
        if occupancy_rate > 0.9:
            competitive_price = min(base_price * 1.3, max_competitor_price * 1.1)
        elif occupancy_rate > 0.7:
            competitive_price = avg_competitor_price * target_position
        elif occupancy_rate < 0.3:
            competitive_price = max(
                min_competitor_price * 0.95,
                self.base_price * 0.6
            )
        else:
            competitive_price = avg_competitor_price * target_position
        
        # Queue pressure adjustment
        if queue_length > 8:
            competitive_price *= 1.15
        elif queue_length > 5:
            competitive_price *= 1.08
        
        # Apply bounds
        min_price = self.base_price * 0.5
        max_price = self.base_price * 2.5
        
        return np.clip(competitive_price, min_price, max_price)
    
    def calculate_all_prices(self, row, competitor_prices=None):
        """Calculate prices using all three models"""
        
        # Current price (start with base price)
        current_price = getattr(row, 'CurrentPrice', self.base_price)
        
        # Model 1: Linear pricing
        price_linear = self.model_1_linear_pricing(
            current_price, row.Occupancy, row.Capacity
        )
        
        # Model 2: Demand-based pricing
        price_demand = self.model_2_demand_based_pricing(
            row.Occupancy, row.Capacity, row.QueueLength,
            row.TrafficLevel, row.IsSpecialDay, row.VehicleType, row.Hour
        )
        
        # Model 3: Competitive pricing
        if competitor_prices is None:
            competitor_prices = [self.base_price * 0.9, self.base_price * 1.1]
        
        price_competitive = self.model_3_competitive_pricing(
            self.base_price, competitor_prices,
            row.Occupancy, row.Capacity, row.QueueLength
        )
        
        return {
            'price_linear': price_linear,
            'price_demand': price_demand,
            'price_competitive': price_competitive,
            'average_price': (price_linear + price_demand + price_competitive) / 3
        }

# =============================================================================
# STREAMING SIMULATION (When Pathway is not available)
# =============================================================================

class StreamingSimulator:
    """
    Simulate real-time streaming for pricing updates
    """
    
    def __init__(self, df, pricing_engine):
        self.df = df
        self.pricing_engine = pricing_engine
        self.current_index = 0
        self.results = []
    
    def process_next_batch(self, batch_size=10):
        """Process next batch of data"""
        if self.current_index >= len(self.df):
            return None
        
        end_index = min(self.current_index + batch_size, len(self.df))
        batch = self.df.iloc[self.current_index:end_index]
        
        batch_results = []
        for _, row in batch.iterrows():
            # Calculate competitor prices (simplified)
            competitor_prices = [
                self.pricing_engine.base_price * (0.8 + 0.4 * np.random.random())
                for _ in range(3)
            ]
            
            # Calculate all prices
            prices = self.pricing_engine.calculate_all_prices(row, competitor_prices)
            
            result = {
                'Timestamp': row.Timestamp,
                'ParkingLotID': row.ParkingLotID,
                'Occupancy': row.Occupancy,
                'Capacity': row.Capacity,
                'OccupancyRate': row.OccupancyRate,
                'QueueLength': row.QueueLength,
                'TrafficLevel': row.TrafficLevel,
                'Hour': row.Hour,
                'IsSpecialDay': row.IsSpecialDay,
                'VehicleType': row.VehicleType,
                **prices
            }
            
            batch_results.append(result)
        
        self.results.extend(batch_results)
        self.current_index = end_index
        
        return pd.DataFrame(batch_results)
    
    def get_all_results(self):
        """Get all processed results"""
        return pd.DataFrame(self.results)

# =============================================================================
# VISUALIZATION AND ANALYSIS
# =============================================================================

def create_comprehensive_analysis(results_df):
    """
    Create comprehensive analysis and visualizations
    """
    
    # Set up the plotting style
    plt.style.use('seaborn-v0_8')
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle('Dynamic Parking Pricing System - Comprehensive Analysis', fontsize=16)
    
    # 1. Price comparison across models
    axes[0, 0].plot(results_df.index, results_df['price_linear'], label='Linear Model', alpha=0.7)
    axes[0, 0].plot(results_df.index, results_df['price_demand'], label='Demand Model', alpha=0.7)
    axes[0, 0].plot(results_df.index, results_df['price_competitive'], label='Competitive Model', alpha=0.7)
    axes[0, 0].plot(results_df.index, results_df['average_price'], label='Average Price', linewidth=2)
    axes[0, 0].set_title('Price Comparison Across Models')
    axes[0, 0].set_xlabel('Time Steps')
    axes[0, 0].set_ylabel('Price ($)')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Occupancy vs Price relationship
    axes[0, 1].scatter(results_df['OccupancyRate'], results_df['price_demand'], 
                      c=results_df['TrafficLevel'], cmap='viridis', alpha=0.6)
    axes[0, 1].set_title('Occupancy Rate vs Demand-Based Price')
    axes[0, 1].set_xlabel('Occupancy Rate')
    axes[0, 1].set_ylabel('Price ($)')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Hourly pricing patterns
    hourly_prices = results_df.groupby('Hour')['average_price'].mean()
    axes[0, 2].bar(hourly_prices.index, hourly_prices.values, alpha=0.7)
    axes[0, 2].set_title('Average Hourly Pricing Patterns')
    axes[0, 2].set_xlabel('Hour of Day')
    axes[0, 2].set_ylabel('Average Price ($)')
    axes[0, 2].grid(True, alpha=0.3)
    
    # 4. Queue length impact
    axes[1, 0].scatter(results_df['QueueLength'], results_df['price_demand'], 
                      c=results_df['OccupancyRate'], cmap='plasma', alpha=0.6)
    axes[1, 0].set_title('Queue Length vs Price (colored by Occupancy)')
    axes[1, 0].set_xlabel('Queue Length')
    axes[1, 0].set_ylabel('Price ($)')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 5. Price distribution
    axes[1, 1].hist(results_df['price_linear'], alpha=0.5, label='Linear', bins=30)
    axes[1, 1].hist(results_df['price_demand'], alpha=0.5, label='Demand', bins=30)
    axes[1, 1].hist(results_df['price_competitive'], alpha=0.5, label='Competitive', bins=30)
    axes[1, 1].set_title('Price Distribution Across Models')
    axes[1, 1].set_xlabel('Price ($)')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    # 6. Revenue estimation
    # Assume demand decreases with price (simple demand curve)
    results_df['estimated_demand'] = 100 * (1 - results_df['average_price'] / (2 * results_df['average_price'].max()))
    results_df['estimated_revenue'] = results_df['average_price'] * results_df['estimated_demand']
    
    axes[1, 2].plot(results_df.index, results_df['estimated_revenue'], color='green', linewidth=2)
    axes[1, 2].set_title('Estimated Revenue Over Time')
    axes[1, 2].set_xlabel('Time Steps')
    axes[1, 2].set_ylabel('Estimated Revenue ($)')
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Summary statistics
    print("\n=== PRICING SYSTEM PERFORMANCE SUMMARY ===")
    print(f"Total time periods analyzed: {len(results_df)}")
    print(f"Average price (Linear Model): ${results_df['price_linear'].mean():.2f}")
    print(f"Average price (Demand Model): ${results_df['price_demand'].mean():.2f}")
    print(f"Average price (Competitive Model): ${results_df['price_competitive'].mean():.2f}")
    print(f"Price volatility (std): ${results_df['average_price'].std():.2f}")
    print(f"Max price observed: ${results_df['average_price'].max():.2f}")
    print(f"Min price observed: ${results_df['average_price'].min():.2f}")
    print(f"Average occupancy rate: {results_df['OccupancyRate'].mean():.2%}")
    print(f"Average queue length: {results_df['QueueLength'].mean():.1f}")
    print(f"Total estimated revenue: ${results_df['estimated_revenue'].sum():.2f}")

# =============================================================================
# MAIN EXECUTION FUNCTION
# =============================================================================

def run_parking_pricing_system(df):
    """
    Main function to run the complete parking pricing system
    """
    
    print("=== DYNAMIC PARKING PRICING SYSTEM ===")
    print("Initializing system...")
    
    # Preprocess data
    df_processed = preprocess_parking_data(df)
    print(f"Processed {len(df_processed)} records")
    
    # Initialize pricing engine
    pricing_engine = AdvancedPricingEngine(base_price=10)
    
    # Create streaming simulator
    simulator = StreamingSimulator(df_processed, pricing_engine)
    
    # Process data in batches
    print("Processing data in real-time simulation...")
    all_results = []
    
    while True:
        batch_result = simulator.process_next_batch(batch_size=50)
        if batch_result is None:
            break
        all_results.append(batch_result)
    
    # Combine all results
    final_results = pd.concat(all_results, ignore_index=True)
    
    # Create comprehensive analysis
    create_comprehensive_analysis(final_results)
    
    return final_results, pricing_engine

# =============================================================================
# EXAMPLE USAGE
# =============================================================================

def create_sample_data():
    """Create sample data for testing"""
    np.random.seed(42)
    n_records = 1000
    
    # Generate timestamps
    start_time = datetime.now() - timedelta(hours=24)
    timestamps = [start_time + timedelta(minutes=i) for i in range(n_records)]
    
    # Generate sample data
    sample_data = {
        'LastUpdatedDate': [t.strftime('%d-%m-%Y') for t in timestamps],
        'LastUpdatedTime': [t.strftime('%H:%M:%S') for t in timestamps],
        'Latitude': np.random.uniform(19.0, 19.2, n_records),
        'Longitude': np.random.uniform(72.8, 73.0, n_records),
        'Occupancy': np.random.randint(0, 100, n_records),
        'Capacity': np.random.randint(80, 120, n_records),
        'QueueLength': np.random.randint(0, 15, n_records),
        'VehicleType': np.random.choice(['Car', 'Bike', 'Truck'], n_records),
        'TrafficConditionNearby': np.random.choice(['Low', 'Medium', 'High'], n_records),
        'IsSpecialDay': np.random.choice([0, 1], n_records, p=[0.8, 0.2])
    }
    
    return pd.DataFrame(sample_data)

# Example usage
if __name__ == "__main__":
    # Create sample data
    sample_df = create_sample_data()
    
    # Run the system
    results, engine = run_parking_pricing_system(sample_df)
    
    print("\nSystem execution completed successfully!")
    print(f"Final results shape: {results.shape}")