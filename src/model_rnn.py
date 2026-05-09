import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import os

def build_rnn(input_shape, units=64, dropout_rate=0.2):
    model = Sequential([
        SimpleRNN(units, input_shape=input_shape, return_sequences=True),
        Dropout(dropout_rate),
        SimpleRNN(units // 2),
        Dropout(dropout_rate),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')  # binary: up or down
    ])
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    model.summary()
    return model

def train_rnn(X_train, y_train, X_val, y_val, epochs=50, batch_size=32):
    input_shape = (X_train.shape[1], X_train.shape[2])
    model = build_rnn(input_shape)
    
    os.makedirs('models', exist_ok=True)
    callbacks = [
        EarlyStopping(patience=10, restore_best_weights=True, verbose=1),
        ModelCheckpoint('models/best_rnn.keras', save_best_only=True, verbose=1)
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

if __name__ == "__main__":
    X_train = np.load('data/processed/X_train.npy')
    y_train = np.load('data/processed/y_train.npy')
    X_val = np.load('data/processed/X_val.npy')
    y_val = np.load('data/processed/y_val.npy')
    
    model, history = train_rnn(X_train, y_train, X_val, y_val)
    print("RNN training complete!")