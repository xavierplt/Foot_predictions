import pandas as pd
import glob
import Config

def fetch_data():
    """
    Loads historical data from local CSV files.
    """
    # Find all CSV files in the data folder
    all_files = glob.glob(Config.DATA_PATH)
    
    if not all_files:
        raise FileNotFoundError(
            "❌ No CSV files found! "
            "Please create a 'data' folder and add CSV files from football-data.co.uk"
        )

    print(f"📂 Loading {len(all_files)} CSV files...")
    matches = []
    
    for filename in all_files:
        try:
            df_csv = pd.read_csv(filename)
            
            # Check for required columns
            # FTHG = Full Time Home Goals, FTAG = Full Time Away Goals
            required_cols = ['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG']
            if not all(col in df_csv.columns for col in required_cols):
                print(f"⚠️ Skipping {filename} (Missing columns)")
                continue

            for _, row in df_csv.iterrows():
                # Ignore matches that haven't been played (missing scores)
                if pd.notna(row['FTHG']) and pd.notna(row['FTAG']):
                    match = {
                        # 'dayfirst=True' is important for European date formats (DD/MM/YYYY)
                        'date': pd.to_datetime(row['Date'], dayfirst=True),
                        'home_team': row['HomeTeam'].strip(),
                        'away_team': row['AwayTeam'].strip(),
                        'home_score': int(row['FTHG']),
                        'away_score': int(row['FTAG'])
                    }
                    matches.append(match)
                    
        except Exception as e:
            print(f"⚠️ Error reading {filename}: {e}")

    df = pd.DataFrame(matches)
    
    if df.empty:
        raise ValueError("❌ CSV files are empty or unreadable.")

    # Chronological sort is crucial for LSTM
    df = df.sort_values('date').reset_index(drop=True)
    
    print(f"✅ Data loaded: {len(df)} matches ready.")
    return df