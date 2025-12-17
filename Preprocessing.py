import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.utils import to_categorical
import Config

class DataProcessor:
    def __init__(self):
        self.scaler = MinMaxScaler()
    
    def calculate_match_features(self, row, team_name):
        is_home = (row['home_team'] == team_name)
        
        goals_for = row['home_score'] if is_home else row['away_score']
        goals_against = row['away_score'] if is_home else row['home_score']
        
        if row['home_score'] == row['away_score']:
            result = 1 
        elif (is_home and row['home_score'] > row['away_score']) or (not is_home and row['away_score'] > row['home_score']):
            result = 2 
        else:
            result = 0 

        features = [
            1 if is_home else 0,
            goals_for,
            goals_against,
            goals_for - goals_against,
            result,
        ] + [0] * (Config.N_FEATURES - 5)
        
        return features, result

    def create_sequences(self, df):
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date')
        
        teams = pd.concat([df['home_team'], df['away_team']]).unique()
        X_sequences, y_labels = [], []

        print("🔄 Generating time sequences...")

        for team in teams:
            team_matches = df[(df['home_team'] == team) | (df['away_team'] == team)].copy()
            
            # --- Correction 1: S'assurer que les données sont float dès le début ---
            team_history = []
            team_results = []
            
            for _, row in team_matches.iterrows():
                feats, res = self.calculate_match_features(row, team)
                team_history.append(feats)
                team_results.append(res)
            
            # Conversion explicite en float32
            data_array = np.array(team_history, dtype=np.float32)

            if len(data_array) > Config.SEQUENCE_LENGTH:
                for i in range(len(data_array) - Config.SEQUENCE_LENGTH):
                    seq = data_array[i : i + Config.SEQUENCE_LENGTH]
                    target = team_results[i + Config.SEQUENCE_LENGTH]
                    
                    # --- Correction 2: Vérification stricte des dimensions ---
                    if seq.shape == (Config.SEQUENCE_LENGTH, Config.N_FEATURES):
                        X_sequences.append(seq)
                        y_labels.append(target)

        # --- Correction 3: Utilisation de np.array avec dtype explicite ---
        X = np.array(X_sequences, dtype=np.float32)
        y = np.array(y_labels, dtype=np.float32)
        
        # Vérification de sécurité
        if len(X) == 0:
            print("⚠️ AVERTISSEMENT: Aucune séquence valide créée !")
            return np.array([]), np.array([])

        # Normalisation
        nsamples, nx, ny = X.shape
        X_2d = X.reshape((nsamples * nx, ny))
        X_2d_scaled = self.scaler.fit_transform(X_2d)
        X = X_2d_scaled.reshape((nsamples, nx, ny))
            
        y = to_categorical(y, num_classes=Config.N_CLASSES)

        return X, y