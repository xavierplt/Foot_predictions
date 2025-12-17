import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Masking
from tensorflow.keras.optimizers import Adam
import Config

def build_lstm_model():
    """
    Builds the LSTM model structure.
    """
    model = Sequential([
        # Masking layer is optional here since we have fixed lengths, but good practice
        Masking(mask_value=-1., input_shape=(Config.SEQUENCE_LENGTH, Config.N_FEATURES)),
        
        # LSTM Layer
        LSTM(Config.LSTM_UNITS, return_sequences=False, activation='tanh'),
        
        # Dropout to prevent overfitting
        Dropout(Config.DROPOUT_RATE),
        
        # Hidden Dense Layer
        Dense(32, activation='relu'),
        
        # Output Layer (3 Classes: Loss, Draw, Win)
        Dense(Config.N_CLASSES, activation='softmax')
    ])
    
    optimizer = Adam(learning_rate=Config.LEARNING_RATE)
    
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model