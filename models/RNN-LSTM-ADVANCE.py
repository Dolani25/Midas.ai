import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Attention, LayerNormalization, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import IsolationForest
import mlflow
import mlflow.tensorflow

# Set up MLflow
mlflow.set_experiment("FOREX_RNN_LSTM_Advanced")

# Data Preparation
def load_and_prepare_data(file_path):
    data = pd.read_csv(file_path)
    data['Date'] = pd.to_datetime(data['Date'])
    data.set_index('Date', inplace=True)
    return data

def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:(i + seq_length)])
        y.append(data[i + seq_length])
    return np.array(X), np.array(y)

# Anomaly Detection
def detect_anomalies(data, contamination=0.01):
    iso_forest = IsolationForest(contamination=contamination, random_state=42)
    anomalies = iso_forest.fit_predict(data)
    return anomalies != -1

# Attention Layer
class AttentionLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)
        
    def build(self, input_shape):
        self.W = self.add_weight(name="att_weight", shape=(input_shape[-1], 1),
                                 initializer="normal")
        self.b = self.add_weight(name="att_bias", shape=(input_shape[1], 1),
                                 initializer="zeros")
        super(AttentionLayer, self).build(input_shape)
        
    def call(self, x):
        et = tf.keras.backend.squeeze(tf.keras.backend.tanh(tf.keras.backend.dot(x, self.W) + self.b), axis=-1)
        at = tf.keras.backend.softmax(et)
        at = tf.keras.backend.expand_dims(at, axis=-1)
        output = x * at
        return tf.keras.backend.sum(output, axis=1)

# Model Architecture
def create_model(seq_length, n_features):
    inputs = Input(shape=(seq_length, n_features))
    x = LSTM(64, return_sequences=True)(inputs)
    x = LayerNormalization()(x)
    x = Dropout(0.2)(x)
    x = LSTM(32, return_sequences=True)(x)
    x = LayerNormalization()(x)
    x = Dropout(0.2)(x)
    x = AttentionLayer()(x)
    outputs = Dense(1)(x)
    model = Model(inputs=inputs, outputs=outputs)
    return model

# Training Function
def train_model(X_train, y_train, X_val, y_val, epochs=100, batch_size=32):
    model = create_model(X_train.shape[1], X_train.shape[2])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    
    callbacks = [
        EarlyStopping(patience=10, restore_best_weights=True),
        ModelCheckpoint('best_model.h5', save_best_only=True)
    ]
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1
    )
    
    return model, history

# Incremental Learning
def incremental_learning(model, X_new, y_new, epochs=10):
    model.fit(X_new, y_new, epochs=epochs, verbose=0)
    return model

# Main Execution
def main():
    with mlflow.start_run():
        # Load and prepare data
        data = load_and_prepare_data('forex_data.csv')
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(data)
        
        # Detect and remove anomalies
        normal_indices = detect_anomalies(scaled_data)
        scaled_data = scaled_data[normal_indices]
        
        # Create sequences
        seq_length = 30
        X, y = create_sequences(scaled_data, seq_length)
        
        # Set up time series cross-validation
        tscv = TimeSeriesSplit(n_splits=5)
        
        for train_index, val_index in tscv.split(X):
            X_train, X_val = X[train_index], X[val_index]
            y_train, y_val = y[train_index], y[val_index]
            
            # Train model
            model, history = train_model(X_train, y_train, X_val, y_val)
            
            # Log metrics
            mlflow.log_metric("val_loss", min(history.history['val_loss']))
            
            # Incremental learning on validation set
            model = incremental_learning(model, X_val, y_val)
        
        # Save the final model
        mlflow.tensorflow.log_model(model, "model")
        
        # Make predictions
        predictions = model.predict(X_val)
        
        # Inverse transform predictions and actual values
        predictions = scaler.inverse_transform(predictions)
        y_val_original = scaler.inverse_transform(y_val.reshape(-1, 1))
        
        # Calculate final RMSE
        rmse = np.sqrt(np.mean((predictions - y_val_original)**2))
        mlflow.log_metric("final_rmse", rmse)
        
        print(f"Final RMSE: {rmse}")

if __name__ == "__main__":
    main()