# --- FILE PATHS ---
DATA_PATH = "data/*.csv"
MODEL_PATH = "saved_models/ligue1_predictor.keras"
SCALER_PATH = "saved_models/scaler.pkl"

# --- DATA GENERATION ---
# We look at the last 10 matches to predict the next one
SEQUENCE_LENGTH = 10    
# Features: [Is_Home, Goals_For, Goals_Against, Goal_Diff, Result]
N_FEATURES = 5          
# Classes: [Loss, Draw, Win]
N_CLASSES = 3           

# --- HYPERPARAMETERS ---
LSTM_UNITS = 128
LEARNING_RATE = 0.001
EPOCHS = 60
BATCH_SIZE = 16
DROPOUT_RATE = 0.2

# --- HYPERPARAMETERS ---
API_KEY = 123
LIGUE_1_ID = 4334