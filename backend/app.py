import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model

# Load the dataset
file_path = 'copper_final.csv'  # Update the path if necessary
data = pd.read_csv(file_path)

# Preprocessing
data['VALID ON'] = pd.to_datetime(data['VALID ON'], format='%d-%m-%Y')
data = data.sort_values('VALID ON')
data['PRICE'] = data['PRICE'].replace(',', '', regex=True).astype(float)

# Visualize the data
plt.figure(figsize=(10, 5))
plt.plot(data['VALID ON'], data['PRICE'], label="Copper Price")
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.title('Copper Prices Over Time')
plt.legend()
plt.show()

# Prepare the data for prediction
prices = data['PRICE'].values.reshape(-1, 1)

# Scale the data using MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_prices = scaler.fit_transform(prices)

# Load the pre-trained LSTM model
model = load_model('Copper_Price_Prediction_LSTM.h5')

# Prepare the last sequence (15 days) to make predictions
sequence_length = 15  # Use 15 days of data to predict the next day
last_sequence = scaled_prices[-sequence_length:]  # Last sequence of 15 days
predicted_prices = []

# Predict the next 3 days' prices
for _ in range(3):  # Predict for 3 days
    last_sequence_expanded = np.expand_dims(last_sequence, axis=0)  # Expand dimensions for the model
    next_day_prediction = model.predict(last_sequence_expanded)
    predicted_prices.append(next_day_prediction[0, 0])
    
    # Update the sequence with the predicted value
    last_sequence = np.append(last_sequence, next_day_prediction, axis=0)
    last_sequence = last_sequence[1:]  # Remove the first element to maintain the sequence length

# Inverse transform the predicted prices
predicted_prices = scaler.inverse_transform(np.array(predicted_prices).reshape(-1, 1))

# Generate future dates
future_dates = pd.date_range(start=data['VALID ON'].iloc[-1] + pd.Timedelta(days=1), periods=3)

# Print the predicted prices with dates
print("Predicted Prices for the Next 3 Days:")
for date, price in zip(future_dates, predicted_prices.flatten()):
    print(f"{date.strftime('%Y-%m-%d')}: ${price:.2f}")