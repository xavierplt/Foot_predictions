import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Masking
from tensorflow.keras.optimizers import Adam
import Config

def build_lstm_model():
    """Builds and compiles the Keras LSTM model."""
    
    model = Sequential([
        # Layer 1: Masking (Handles missing values/padding if necessary)
        Masking(mask_value=-1., input_shape=(Config.SEQUENCE_LENGTH, Config.N_FEATURES)),
        
        # Layer 2: LSTM Core
        # return_sequences=False because we only want the final output after analyzing the sequence
        LSTM(Config.LSTM_UNITS, return_sequences=False, activation='tanh'),
        
        # Layer 3: Dropout (Prevents Overfitting)
        Dropout(Config.DROPOUT_RATE),
        
        # Layer 4: Hidden Dense Layer
        Dense(32, activation='relu'),
        
        # Layer 5: Output Layer (Softmax for probability distribution across 3 classes)
        Dense(Config.N_CLASSES, activation='softmax')
    ])

    optimizer = Adam(learning_rate=Config.LEARNING_RATE)

    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model