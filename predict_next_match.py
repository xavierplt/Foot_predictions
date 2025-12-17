# predict.py
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
import pickle
import Config
from Data_loader import fetch_data
from Preprocessing import DataProcessor

def get_last_n_matches(df, team_name, n, processor):
    """Retrieves and formats the last N matches for a specific team."""
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')
    
    # Filter matches for this team
    team_matches = df[(df['home_team'] == team_name) | (df['away_team'] == team_name)].copy()
    
    if len(team_matches) < n:
        print(f"⚠️ Not enough history for {team_name} (Found: {len(team_matches)}, Required: {n})")
        return None
        
    # Get the last N matches
    last_matches = team_matches.tail(n)
    
    formatted_sequence = []
    for _, row in last_matches.iterrows():
        feats, _ = processor.calculate_match_features(row, team_name)
        formatted_sequence.append(feats)
        
    return np.array(formatted_sequence, dtype=np.float32)

def predict_match(home_team_name, away_team_name):
    print(f"\n🔮 PREDICTION : {home_team_name} (Home) vs {away_team_name} (Away)")
    
    # 1. Load Model and Scaler
    try:
        model = load_model(Config.MODEL_PATH)
        with open(Config.SCALER_PATH, 'rb') as f:
            scaler = pickle.load(f)
    except Exception as e:
        print(f"❌ Error loading model/scaler: {e}")
        print("Make sure you ran 'train.py' first.")
        return

    # 2. Load latest data
    df = fetch_data()
    processor = DataProcessor()
    
    # 3. Prepare Home Team Data
    last_10_home = get_last_n_matches(df, home_team_name, Config.SEQUENCE_LENGTH, processor)
    
    if last_10_home is None:
        return

    # Normalize using the loaded scaler
    # We must reshape to 2D -> transform -> reshape back to 3D
    seq_home_2d = last_10_home.reshape(Config.SEQUENCE_LENGTH, Config.N_FEATURES)
    seq_home_scaled = scaler.transform(seq_home_2d)
    seq_home_final = seq_home_scaled.reshape(1, Config.SEQUENCE_LENGTH, Config.N_FEATURES)
    
    # 4. Predict
    pred_probs = model.predict(seq_home_final, verbose=0)[0]

    # 5. Display Results
    # Index 0: Loss (Away Win), 1: Draw, 2: Win (Home Win)
    win_prob = pred_probs[2] * 100
    draw_prob = pred_probs[1] * 100
    loss_prob = pred_probs[0] * 100

    print("-" * 30)
    print(f"📊 Form Analysis for {home_team_name}:")
    print(f"  Win Probability  : {win_prob:.1f}%")
    print(f"  Draw Probability : {draw_prob:.1f}%")
    print(f"  Loss Probability : {loss_prob:.1f}%")
    print("-" * 30)
    
    winner_idx = np.argmax(pred_probs)
    verdicts = ["Away Win", "Draw", "Home Win"]
    print(f"🏆 AI VERDICT: {verdicts[winner_idx]}")

if __name__ == "__main__":
    # CHANGE TEAM NAMES HERE TO MATCH YOUR CSV FILE EXACTLY
    predict_match("Paris SG", "Marseille")