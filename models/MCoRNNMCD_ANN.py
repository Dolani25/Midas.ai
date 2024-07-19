import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import logging
import os
import sys
from datetime import datetime

# Set up logging
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f"mcornnmcd_ann_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(sys.stdout)
    ]
)

# Custom layer for Monte Carlo Dropout
class MCDropout(layers.Dropout):
    def call(self, inputs):
        return super().call(inputs, training=True)

# Orthogonal Initializer with Gain
def orthogonal_initializer(gain=1.0):
    return tf.keras.initializers.Orthogonal(gain)

# HSwish Activation Function
def hswish(x, beta=0.5):
    return x * tf.nn.relu6(x + 3) / 6 * beta

# CoRNNMCD Module
def create_cornnmcd(input_shape):
    inputs = keras.Input(shape=input_shape)
    x = layers.Conv1D(64, kernel_size=3, padding='same', activation='relu', kernel_initializer=orthogonal_initializer())(inputs)
    x = layers.LSTM(128, return_sequences=True, kernel_initializer=orthogonal_initializer())(x)
    x = MCDropout(0.3)(x)
    return keras.Model(inputs, x)

# CoGRUMCD Module
def create_cogrumcd(input_shape):
    inputs = keras.Input(shape=input_shape)
    x = layers.Conv1D(64, kernel_size=3, padding='same', activation='relu', kernel_initializer=orthogonal_initializer())(inputs)
    x = layers.GRU(128, return_sequences=True, kernel_initializer=orthogonal_initializer())(x)
    x = MCDropout(0.3)(x)
    return keras.Model(inputs, x)

# MCoRNNMCD-ANN Model
def create_mcornnmcd_ann(price_input_shape, sentiment_input_shape):
    price_input = keras.Input(shape=price_input_shape)
    sentiment_input = keras.Input(shape=sentiment_input_shape)

    cornnmcd = create_cornnmcd(price_input_shape)
    cogrumcd = create_cogrumcd(sentiment_input_shape)

    price_features = cornnmcd(price_input)
    sentiment_features = cogrumcd(sentiment_input)

    x = layers.Concatenate()([price_features, sentiment_features])
    x = layers.Flatten()(x)

    x = layers.Dense(256, activation=hswish)(x)
    x = layers.Dense(128, activation=hswish)(x)
    x = layers.Dense(64, activation=hswish)(x)

    output = layers.Dense(1)(x)

    model = keras.Model(inputs=[price_input, sentiment_input], outputs=output)
    return model

# Data Preparation Function
def prepare_data(price_data, sentiment_data, sequence_length):
    try:
        X_price = []
        X_sentiment = []
        y = []

        for i in range(len(price_data) - sequence_length):
            X_price.append(price_data[i:i+sequence_length])
            X_sentiment.append(sentiment_data[i:i+sequence_length])
            y.append(price_data[i+sequence_length])

        return np.array(X_price), np.array(X_sentiment), np.array(y)
    except Exception as e:
        logging.error(f"Error in data preparation: {str(e)}")
        raise

# Data Loading Function
def load_data(price_file, sentiment_file):
    try:
        price_data = pd.read_csv(price_file)
        sentiment_data = pd.read_csv(sentiment_file)
        logging.info(f"Data loaded successfully. Price data shape: {price_data.shape}, Sentiment data shape: {sentiment_data.shape}")
        return price_data, sentiment_data
    except FileNotFoundError as e:
        logging.error(f"File not found: {str(e)}")
        raise
    except Exception as e:
        logging.error(f"Error loading data: {str(e)}")
        raise

# Main execution
def main():
    try:
        # Load data
        price_data, sentiment_data = load_data('price_data.csv', 'sentiment_data.csv')
        
        sequence_length = 60  # Adjust based on your requirements

        X_price, X_sentiment, y = prepare_data(price_data.values, sentiment_data.values, sequence_length)

        # Split the data
        X_price_train, X_price_test, X_sentiment_train, X_sentiment_test, y_train, y_test = train_test_split(
            X_price, X_sentiment, y, test_size=0.2, random_state=42)

        # Normalize the data
        scaler = StandardScaler()
        X_price_train = scaler.fit_transform(X_price_train.reshape(-1, X_price_train.shape[-1])).reshape(X_price_train.shape)
        X_price_test = scaler.transform(X_price_test.reshape(-1, X_price_test.shape[-1])).reshape(X_price_test.shape)
        X_sentiment_train = scaler.fit_transform(X_sentiment_train.reshape(-1, X_sentiment_train.shape[-1])).reshape(X_sentiment_train.shape)
        X_sentiment_test = scaler.transform(X_sentiment_test.reshape(-1, X_sentiment_test.shape[-1])).reshape(X_sentiment_test.shape)

        logging.info("Data preprocessing completed.")

        # Create and compile the model
        model = create_mcornnmcd_ann((sequence_length, X_price_train.shape[-1]), (sequence_length, X_sentiment_train.shape[-1]))
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])

        logging.info("Model created and compiled.")

        # Train the model
        history = model.fit(
            [X_price_train, X_sentiment_train], y_train,
            validation_data=([X_price_test, X_sentiment_test], y_test),
            epochs=100, batch_size=32,
            callbacks=[
                keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
                keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5)
            ]
        )

        logging.info("Model training completed.")

        # Evaluate the model
        test_loss, test_mae = model.evaluate([X_price_test, X_sentiment_test], y_test)
        logging.info(f"Test Loss: {test_loss}, Test MAE: {test_mae}")

        # Make predictions
        predictions = model.predict([X_price_test, X_sentiment_test])

        # Save the model
        model.save('mcornnmcd_ann_model.h5')
        logging.info("Model saved successfully.")

    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.error(f"Program terminated due to error: {str(e)}")
        sys.exit(1)