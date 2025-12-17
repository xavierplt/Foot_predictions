import Config
from Data_loader import fetch_ligue1_data
from Preprocessing import DataProcessor
from Model import build_lstm_model
from tensorflow.keras.callbacks import EarlyStopping
import os

def main():
    # 1. Load Data
    df = fetch_ligue1_data()
    if df.empty:
        print("❌ No data retrieved. Aborting.")
        return

    # 2. Preprocessing
    processor = DataProcessor()
    X, y = processor.create_sequences(df)
    
    print("-" * 30)
    print(f"📊 DEBUG SHAPES:")
    print(f"X shape: {X.shape}") 
    print(f"y shape: {y.shape}")
    print(f"X dtype: {X.dtype}")
    print("-" * 30)
    
    # Vérification critique avant de continuer
    if X.ndim != 3:
        raise ValueError(f"❌ ERREUR CRITIQUE: X doit être en 3D (Samples, {Config.SEQUENCE_LENGTH}, {Config.N_FEATURES}), mais reçu {X.shape}")
    
    # Split Train/Test (80% / 20%)
    split_idx = int(0.8 * len(X))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    # 3. Model Building
    model = build_lstm_model()
    model.summary()

    # 4. Training
    print("🚀 Starting training on Ligue 1 data...")
    callbacks = [EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)]
    
    history = model.fit(
        X_train, y_train,
        epochs=Config.EPOCHS,
        batch_size=Config.BATCH_SIZE,
        validation_data=(X_test, y_test),
        callbacks=callbacks
    )

    # 5. Saving
    if not os.path.exists('saved_models'):
        os.makedirs('saved_models')
    
    model.save('saved_models/ligue1_predictor.keras')
    print("💾 Model saved successfully to 'saved_models/ligue1_predictor.keras'")

if __name__ == "__main__":
    main()