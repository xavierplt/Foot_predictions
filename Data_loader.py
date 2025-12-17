import requests
import pandas as pd
import numpy as np
import Config

def fetch_ligue1_data():
    """
    Fetches Ligue 1 match data for the seasons defined in config.py.
    Returns: pd.DataFrame containing match history.
    """
    all_matches = []
    print(f"📡 Connecting to TheSportsDB (Seasons: {Config.SEASONS})...")

    for season in Config.SEASONS:
        url = f"https://www.thesportsdb.com/api/v1/json/{Config.API_KEY}/eventsseason.php?id={Config.LIGUE_1_ID}&s={season}"
        try:
            response = requests.get(url)
            data = response.json()
            
            if data['events']:
                for event in data['events']:
                    # We only keep matches that have actually been played (have a score)
                    if event['intHomeScore'] is not None:
                        match = {
                            'date': event['dateEvent'],
                            'home_team': event['strHomeTeam'],
                            'away_team': event['strAwayTeam'],
                            'home_score': int(event['intHomeScore']),
                            'away_score': int(event['intAwayScore']),
                            # NOTE: The free API does not provide possession/shots data.
                            # We simulate them here for the sake of the exercise.
                            'home_shots': np.random.randint(5, 20),
                            'away_shots': np.random.randint(5, 20),
                        }
                        all_matches.append(match)
        except Exception as e:
            print(f"⚠️ Error fetching season {season}: {e}")

    df = pd.DataFrame(all_matches)
    print(f"✅ Data loaded: {len(df)} matches found.")
    return df