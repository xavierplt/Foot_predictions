import Config
from Data_loader import fetch_data
from Preprocessing import DataProcessor
from Model import build_lstm_model
from tensorflow.keras.callbacks import EarlyStopping
import os

def main():
    # 1. Load Data
    df = fetch_data()
    
    # 2. Preprocessing
    processor = DataProcessor()
    # We set save_scaler=True because this is the training phase
    X, y = processor.create_sequences(df, save_scaler=True)
    
    print("-" * 30)
    print(f"📊 DATA SHAPES:")
    print(f"X: {X.shape}") 
    print(f"y: {y.shape}")
    print("-" * 30)
    
    if len(X) == 0:
        print("❌ Error: Not enough data to create sequences. Add more seasons.")
        return

    # Split Train/Test (80% / 20%)
    split_idx = int(0.8 * len(X))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    # 3. Build Model
    model = build_lstm_model()
    model.summary()

    # 4. Train
    print("🚀 Starting training...")
    callbacks = [EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)]
    
    history = model.fit(
        X_train, y_train,
        epochs=Config.EPOCHS,
        batch_size=Config.BATCH_SIZE,
        validation_data=(X_test, y_test),
        callbacks=callbacks
    )

    # 5. Save Model
    if not os.path.exists('saved_models'):
        os.makedirs('saved_models')
    
    model.save(Config.MODEL_PATH)
    print(f"💾 Model saved successfully to '{Config.MODEL_PATH}'")

if __name__ == "__main__":
    main()