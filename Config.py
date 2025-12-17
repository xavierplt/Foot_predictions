# --- API Settings ---
API_KEY = "123"  # Public test key from TheSportsDB
LIGUE_1_ID = "4334" # ID for French Ligue 1
SEASONS = ["2021-2022", "2022-2023", "2023-2024"]

# --- Data Generation Settings ---
SEQUENCE_LENGTH = 10    # Number of past matches to look at
N_FEATURES = 15         # Number of features per match
N_CLASSES = 3           # 0: Loss, 1: Draw, 2: Win

# --- Model Hyperparameters ---
LSTM_UNITS = 64
LEARNING_RATE = 0.001
EPOCHS = 50
BATCH_SIZE = 16
DROPOUT_RATE = 0.3