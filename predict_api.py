import requests
import pandas as pd
import numpy as np
import pickle
from tensorflow.keras.models import load_model
import Config

# --- API ConfigURATION ---
# Endpoint to get the next 15 matches for a league
NEXT_EVENTS_URL = f"https://www.thesportsdb.com/api/v1/json/{Config.API_KEY}/eventsnextleague.php?id={Config.LIGUE_1_ID}"
# Endpoint to get the last 5/10 matches for a specific team
LAST_EVENTS_URL = "https://www.thesportsdb.com/api/v1/json/{}/eventslast.php?id={}"

def get_upcoming_matches():
    """
    Fetches the list of upcoming matches from TheSportsDB.
    """
    print(f"📡 Fetching upcoming matches from API...")
    try:
        response = requests.get(NEXT_EVENTS_URL)
        data = response.json()
        
        if not data or 'events' not in data or data['events'] is None:
            print("⚠️ API Warning: No upcoming matches found. (Limit of free API key '3').")
            return []
            
        return data['events']
    except Exception as e:
        print(f"❌ Connection Error: {e}")
        return []

def get_team_history_from_api(team_id):
    """
    Fetches the last 10 matches for a specific team via API.
    """
    url = LAST_EVENTS_URL.format(Config.API_KEY, team_id)
    response = requests.get(url)
    data = response.json()
    
    if not data or 'results' not in data or data['results'] is None:
        return None
        
    matches = []
    # The API returns oldest to newest usually, ensuring we sort by date is safer
    results = sorted(data['results'], key=lambda x: x['dateEvent'])
    
    # We need the last 10 matches
    # Note: API might return only 5 matches on free tier. 
    # If < 10, the model might crash or perform poorly. We handle this below.
    last_results = results[-Config.SEQUENCE_LENGTH:] 
    
    for event in last_results:
        # Construct the features exactly like in Preprocessing.py
        # We need to determine if the team was Home or Away in that history match
        is_home_history = (event['idHomeTeam'] == team_id)
        
        home_score = int(event['intHomeScore'])
        away_score = int(event['intAwayScore'])
        
        goals_for = home_score if is_home_history else away_score
        goals_against = away_score if is_home_history else home_score
        
        # Result (0: Loss, 1: Draw, 2: Win)
        if home_score == away_score:
            res = 1
        elif (is_home_history and home_score > away_score) or (not is_home_history and away_score > home_score):
            res = 2
        else:
            res = 0
            
        # Feature Vector [Is_Home, GF, GA, Diff, Res]
        features = [
            1.0 if is_home_history else 0.0,
            float(goals_for),
            float(goals_against),
            float(goals_for - goals_against),
            float(res)
        ]
        matches.append(features)
        
    return np.array(matches, dtype=np.float32)

def predict_upcoming_day():
    # 1. Load Model and Scaler
    print("📂 Loading model and scaler...")
    try:
        model = load_model(Config.MODEL_PATH)
        with open(Config.SCALER_PATH, 'rb') as f:
            scaler = pickle.load(f)
    except Exception as e:
        print(f"❌ Error: {e}. Did you run 'train.py'?")
        return

    # 2. Get Upcoming Fixtures
    fixtures = get_upcoming_matches()
    if not fixtures: return

    print(f"\n🔮 PREDICTIONS FOR UPCOMING GAMES ({len(fixtures)} matches)")
    print("=" * 60)

    for match in fixtures:
        home_name = match['strHomeTeam']
        away_name = match['strAwayTeam']
        home_id = match['idHomeTeam']
        
        # 3. Get History for the HOME team (to analyze their form)
        # (Ideally we could analyze both, but let's start with Home Team perspective)
        history = get_team_history_from_api(home_id)
        
        # Check if we have enough data (SEQUENCE_LENGTH = 10)
        if history is None or len(history) < Config.SEQUENCE_LENGTH:
            print(f"⚠️ Skipping {home_name} vs {away_name}: Not enough history data (Found {len(history) if history is not None else 0})")
            continue
            
        # 4. Preprocessing (Normalization)
        # Reshape to 2D -> Transform -> Reshape to 3D
        seq_2d = history.reshape(Config.SEQUENCE_LENGTH, Config.N_FEATURES)
        seq_scaled = scaler.transform(seq_2d)
        seq_final = seq_scaled.reshape(1, Config.SEQUENCE_LENGTH, Config.N_FEATURES)
        
        # 5. Prediction
        prediction = model.predict(seq_final, verbose=0)[0]
        
        # 6. Display
        prob_win = prediction[2] * 100
        prob_draw = prediction[1] * 100
        prob_loss = prediction[0] * 100
        
        winner_idx = np.argmax(prediction)
        verdicts = [f"Victoire {away_name}", "Match Nul", f"Victoire {home_name}"]
        color = "\033[92m" if winner_idx == 2 else "\033[91m" # Green for Home Win, Red otherwise
        reset = "\033[0m"

        print(f"{home_name} vs {away_name}")
        print(f"   📊 Probas: {home_name} {prob_win:.1f}% | Nul {prob_draw:.1f}% | {away_name} {prob_loss:.1f}%")
        print(f"   🏆 Verdict: {color}{verdicts[winner_idx]}{reset}")
        print("-" * 60)

if __name__ == "__main__":
    predict_upcoming_day()