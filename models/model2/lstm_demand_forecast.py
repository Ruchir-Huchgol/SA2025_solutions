# LSTM-Based Demand Forecasting for Urban Parking Lots
#Section 1: Setup
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.callbacks import EarlyStopping
from sklearn.metrics import mean_squared_error
from math import sqrt

#Section 2: Load Data
df = pd.read_csv("dataset.csv")
df['datetime'] = pd.to_datetime(df['timestamp'])
df.sort_values(by='datetime', inplace=True)

#Section 3: Preprocess Data
lot_df = df[df['lot_id'] == 1].copy()
lot_df.set_index('datetime', inplace=True)
scaler = MinMaxScaler()
lot_df['occupancy_scaled'] = scaler.fit_transform(lot_df[['occupancy']])

#Section 4: Create Sequences
def create_sequences(data, window=10):
    X, y = [], []
    for i in range(len(data) - window):
        X.append(data[i:i+window])
        y.append(data[i+window])
    return np.array(X), np.array(y)
window_size = 10
sequence_data = lot_df['occupancy_scaled'].values
X, y = create_sequences(sequence_data, window=window_size)

X = X.reshape((X.shape[0], X.shape[1], 1))

#Section 5: Split Dataset
split = int(len(X) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

#Section 6: Build LSTM Model
model = Sequential()
model.add(LSTM(64, input_shape=(X.shape[1], 1)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')

early_stop = EarlyStopping(monitor='val_loss', patience=5)

#Section 7: Train Model
history = model.fit(X_train, y_train,
                    validation_data=(X_test, y_test),
                    epochs=50,
                    batch_size=32,
                    callbacks=[early_stop],
                    verbose=1)

#Section 8: Predict and Invert Scaling
y_pred = model.predict(X_test)
y_pred_inv = scaler.inverse_transform(y_pred)
y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1))

# Section 9: Evaluate Performance
rmse = sqrt(mean_squared_error(y_test_inv, y_pred_inv))
print(f"LSTM RMSE: {rmse:.2f}")

#Section 10: Plot Predictions
plt.figure(figsize=(12, 6))
plt.plot(y_test_inv, label='Actual')
plt.plot(y_pred_inv, label='Predicted')
plt.title('LSTM Occupancy Forecast (Lot 1)')
plt.xlabel('Time Steps')
plt.ylabel('Occupancy')
plt.legend()
plt.grid()
plt.show()

#Section 11: Forecast Future Steps
def forecast_future(model, last_sequence, steps=5):
    predictions = []
    input_seq = last_sequence.copy()
    for _ in range(steps):
        pred = model.predict(input_seq.reshape(1, window_size, 1))
        predictions.append(pred[0][0])
        input_seq = np.append(input_seq[1:], pred, axis=0)
    return scaler.inverse_transform(np.array(predictions).reshape(-1, 1))

future_forecast = forecast_future(model, X_test[-1], steps=5)
print("Next 5-Step Forecast:")
print(future_forecast)


