import numpy as np
import mlflow
import mlflow.keras
import os
from sklearn.metrics import accuracy_score, f1_score
from model_rnn import build_rnn, train_rnn
from model_lstm import build_lstm, train_lstm
from model_gru import build_gru, train_gru

# Load data
X_train = np.load('data/processed/X_train.npy')
y_train = np.load('data/processed/y_train.npy')
X_val = np.load('data/processed/X_val.npy')
y_val = np.load('data/processed/y_val.npy')
X_test = np.load('data/processed/X_test.npy')
y_test = np.load('data/processed/y_test.npy')

mlflow.set_tracking_uri("mlruns")
mlflow.set_experiment("Market_Prediction")

def run_experiment(model_type, units, dropout, epochs, batch_size):
    with mlflow.start_run(run_name=f"{model_type}_u{units}_d{dropout}"):
        # Log parameters
        mlflow.log_param("model_type", model_type)
        mlflow.log_param("units", units)
        mlflow.log_param("dropout_rate", dropout)
        mlflow.log_param("epochs", epochs)
        mlflow.log_param("batch_size", batch_size)
        mlflow.log_param("sequence_length", X_train.shape[1])
        
        input_shape = (X_train.shape[1], X_train.shape[2])
        
        # Build and train
        if model_type == "RNN":
            model = build_rnn(input_shape, units=units, dropout_rate=dropout)
        elif model_type == "LSTM":
            model = build_lstm(input_shape, units=units, dropout_rate=dropout)
        elif model_type == "GRU":
            model = build_gru(input_shape, units=units, dropout_rate=dropout)
        
        from tensorflow.keras.callbacks import EarlyStopping
        callbacks = [EarlyStopping(patience=10, restore_best_weights=True)]
        
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=0
        )
        
        # Evaluate
        y_pred = (model.predict(X_test) > 0.5).astype(int).flatten()
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        # Log metrics
        mlflow.log_metric("test_accuracy", acc)
        mlflow.log_metric("test_f1_score", f1)
        mlflow.log_metric("best_val_accuracy", max(history.history['val_accuracy']))
        mlflow.log_metric("epochs_trained", len(history.history['loss']))
        
        # Save model
        mlflow.keras.log_model(model, f"{model_type}_model")
        
        print(f"{model_type} | Accuracy: {acc:.4f} | F1: {f1:.4f}")

# Run experiments for all 3 models
for model_type in ["RNN", "LSTM", "GRU"]:
    run_experiment(model_type, units=64, dropout=0.2, epochs=50, batch_size=32)
    run_experiment(model_type, units=128, dropout=0.3, epochs=50, batch_size=16)

print("\nAll MLflow experiments complete!")
print("Run 'mlflow ui' in terminal to see the dashboard")