import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.utils import to_categorical
import pickle
import os
import Config

class DataProcessor:
    def __init__(self):
        # We use a fixed range (0-1) because football scores are generally low (0-10)
        self.scaler = MinMaxScaler(feature_range=(0, 1))
    
    def calculate_match_features(self, row, team_name):
        """
        Extracts only the essential form features.
        """
        is_home = (row['home_team'] == team_name)
        
        goals_for = row['home_score'] if is_home else row['away_score']
        goals_against = row['away_score'] if is_home else row['home_score']
        
        # Determine Result (0: Loss, 1: Draw, 2: Win)
        if row['home_score'] == row['away_score']:
            result = 1 # Draw
        elif (is_home and row['home_score'] > row['away_score']) or (not is_home and row['away_score'] > row['home_score']):
            result = 2 # Win
        else:
            result = 0 # Loss

        # THE 5 FEATURES
        features = [
            1.0 if is_home else 0.0,          # 1. Home/Away
            float(goals_for),                 # 2. Goals Scored
            float(goals_against),             # 3. Goals Conceded
            float(goals_for - goals_against), # 4. Goal Difference
            float(result)                     # 5. Result
        ]
        
        return features, result

    def create_sequences(self, df, save_scaler=False):
        """
        Generates the (Samples, TimeSteps, Features) matrix.
        """
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date')
        
        teams = pd.concat([df['home_team'], df['away_team']]).unique()
        X_sequences, y_labels = [], []

        print(f"🔄 Processing team form sequences...")

        for team in teams:
            team_matches = df[(df['home_team'] == team) | (df['away_team'] == team)].copy()
            
            team_history = []
            team_results = []
            
            for _, row in team_matches.iterrows():
                feats, res = self.calculate_match_features(row, team)
                team_history.append(feats)
                team_results.append(res)
            
            data_array = np.array(team_history, dtype=np.float32)

            # Create Sliding Windows
            if len(data_array) > Config.SEQUENCE_LENGTH:
                for i in range(len(data_array) - Config.SEQUENCE_LENGTH):
                    # Input (X): The previous matches
                    seq = data_array[i : i + Config.SEQUENCE_LENGTH]
                    # Target (y): The result of the NEXT match
                    target = team_results[i + Config.SEQUENCE_LENGTH]
                    
                    if seq.shape == (Config.SEQUENCE_LENGTH, Config.N_FEATURES):
                        X_sequences.append(seq)
                        y_labels.append(target)

        X = np.array(X_sequences, dtype=np.float32)
        y = np.array(y_labels, dtype=np.float32)

        if len(X) == 0:
            return np.array([]), np.array([])

        # Normalization
        # Reshape to 2D for scaler, then back to 3D
        nsamples, nx, ny = X.shape
        X_2d = X.reshape((nsamples * nx, ny))
        
        if save_scaler:
            X_2d_scaled = self.scaler.fit_transform(X_2d)
            # Save scaler for future predictions
            if not os.path.exists('saved_models'): os.makedirs('saved_models')
            with open(Config.SCALER_PATH, 'wb') as f:
                pickle.dump(self.scaler, f)
            print(f"✅ Scaler saved to '{Config.SCALER_PATH}'")
        else:
            X_2d_scaled = self.scaler.transform(X_2d)

        X = X_2d_scaled.reshape((nsamples, nx, ny))
        y = to_categorical(y, num_classes=Config.N_CLASSES)

        return X, y