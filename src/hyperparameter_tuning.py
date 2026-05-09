import numpy as np
import mlflow
import mlflow.keras
from sklearn.metrics import accuracy_score, f1_score
from model_lstm import build_lstm
from tensorflow.keras.callbacks import EarlyStopping

X_train = np.load('data/processed/X_train.npy')
y_train = np.load('data/processed/y_train.npy')
X_val = np.load('data/processed/X_val.npy')
y_val = np.load('data/processed/y_val.npy')
X_test = np.load('data/processed/X_test.npy')
y_test = np.load('data/processed/y_test.npy')

mlflow.set_tracking_uri("mlruns")
mlflow.set_experiment("Market_Prediction_Tuning")

# Different combinations to try
param_grid = [
    {'units': 32,  'dropout': 0.1, 'batch_size': 16},
    {'units': 64,  'dropout': 0.2, 'batch_size': 32},
    {'units': 128, 'dropout': 0.3, 'batch_size': 32},
    {'units': 256, 'dropout': 0.4, 'batch_size': 64},
]

input_shape = (X_train.shape[1], X_train.shape[2])
best_acc = 0
best_params = None

for params in param_grid:
    with mlflow.start_run(run_name=f"LSTM_tune_u{params['units']}"):
        mlflow.log_params(params)
        
        model = build_lstm(input_shape, 
                          units=params['units'], 
                          dropout_rate=params['dropout'])
        
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=30,
            batch_size=params['batch_size'],
            callbacks=[EarlyStopping(patience=5, restore_best_weights=True)],
            verbose=0
        )
        
        y_pred = (model.predict(X_test) > 0.5).astype(int).flatten()
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        mlflow.log_metric("test_accuracy", acc)
        mlflow.log_metric("test_f1", f1)
        
        print(f"Units={params['units']}, Dropout={params['dropout']} → Acc={acc:.4f}, F1={f1:.4f}")
        
        if acc > best_acc:
            best_acc = acc
            best_params = params

print(f"\nBest params: {best_params} with accuracy {best_acc:.4f}")