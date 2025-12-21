import requests
import pandas as pd
import numpy as np
import pickle
from tensorflow.keras.models import load_model
import Config
from Data_loader import fetch_data
from Preprocessing import DataProcessor

# --- CONFIGURATION ---
# We use the API only to know WHICH matches are coming up next
NEXT_EVENTS_URL = f"https://www.thesportsdb.com/api/v1/json/{Config.API_KEY}/eventsnextleague.php?id={Config.LIGUE_1_ID}"

# Name mapping dictionary (API Name -> CSV Name)
# Fill this in if you see "Not enough history" errors
NAME_MAPPING = {
    "Paris Saint-Germain": "Paris SG",
    "St Etienne": "St_Etienne",
    "Olympique Marseille": "Marseille",
    "Olympique Lyonnais": "Lyon",
    # Add other corrections here based on your CSV files
}

def get_upcoming_schedule():
    """Retrieves the list of upcoming matches via the API"""
    print(f"📡 Retrieving schedule via API...")
    try:
        response = requests.get(NEXT_EVENTS_URL)
        data = response.json()
        if not data or 'events' not in data or data['events'] is None:
            print("⚠️ API: No upcoming matches found (Free key limit or end of season).")
            # For testing, you can uncomment the line below to force a match:
            # return [{'strHomeTeam': 'Paris SG', 'strAwayTeam': 'Marseille'}] 
            return []
        return data['events']
    except Exception as e:
        print(f"❌ API Connection Error: {e}")
        return []

def get_last_n_matches_from_csv(df, team_name_api, n, processor):
    """
    Searches for history in the local CSV instead of the API.
    """
    # 1. Handling name differences (API vs CSV)
    team_name = NAME_MAPPING.get(team_name_api, team_name_api)
    
    # 2. Filtering inside the loaded DataFrame
    # We look for the team either as home or away
    team_matches = df[(df['home_team'] == team_name) | (df['away_team'] == team_name)].copy()
    
    # 3. Checking data quantity
    if len(team_matches) < n:
        print(f"⚠️  Warning: Name '{team_name}' (API: {team_name_api}) not found or not enough matches in CSV.")
        return None

    # 4. Taking the last N matches
    # The df DataFrame is assumed to be already sorted by date by fetch_data()
    last_matches = team_matches.tail(n)
    
    formatted_sequence = []
    
    # 5. Feature Calculation (Exactly as done during training)
    for _, row in last_matches.iterrows():
        feats, _ = processor.calculate_match_features(row, team_name)
        formatted_sequence.append(feats)
        
    return np.array(formatted_sequence, dtype=np.float32)

def main():
    # 1. Loading tools
    print("📂 Loading model, scaler, and CSV data...")
    try:
        model = load_model(Config.MODEL_PATH)
        with open(Config.SCALER_PATH, 'rb') as f:
            scaler = pickle.load(f)
        
        # We load the ENTIRE CSV history into memory just once
        df_history = fetch_data()
        processor = DataProcessor()
        
    except Exception as e:
        print(f"❌ Loading Error: {e}")
        return

    # 2. Retrieving API schedule
    fixtures = get_upcoming_schedule()
    if not fixtures: return

    print(f"\n🔮 UPCOMING MATCH PREDICTIONS ({len(fixtures)} matches)")
    print("=" * 60)

    for match in fixtures:
        home_name_api = match['strHomeTeam']
        away_name_api = match['strAwayTeam']
        
        # 3. Retrieving history from CSV (More reliable!)
        history = get_last_n_matches_from_csv(df_history, home_name_api, Config.SEQUENCE_LENGTH, processor)
        
        if history is None:
            continue
            
        # 4. Normalization (Same as training)
        # Reshape 2D -> Scaler -> Reshape 3D
        seq_2d = history.reshape(Config.SEQUENCE_LENGTH, Config.N_FEATURES)
        seq_scaled = scaler.transform(seq_2d)
        seq_final = seq_scaled.reshape(1, Config.SEQUENCE_LENGTH, Config.N_FEATURES)
        
        # 5. Prediction
        pred = model.predict(seq_final, verbose=0)[0]
        
        # 6. Display/Output
        win_prob = pred[2] * 100
        draw_prob = pred[1] * 100
        loss_prob = pred[0] * 100 # Home Loss = Away Win
        
        winner_idx = np.argmax(pred)
        verdicts = [f"{away_name_api} Win", "Draw", f"{home_name_api} Win"]
        
        # Colors for console output
        color = "\033[92m" if winner_idx == 2 else "\033[91m"
        reset = "\033[0m"

        print(f"⚽ {home_name_api} vs {away_name_api}")
        print(f"   📊 Form (based on last 10 CSV matches):")
        print(f"      {home_name_api} : {win_prob:.1f}%")
        print(f"      Draw           : {draw_prob:.1f}%")
        print(f"      {away_name_api} : {loss_prob:.1f}%")
        print(f"   🏆 AI Verdict: {color}{verdicts[winner_idx]}{reset}")
        print("-" * 60)

if __name__ == "__main__":
    main()