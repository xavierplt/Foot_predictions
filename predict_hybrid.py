import requests
import pandas as pd
import numpy as np
import pickle
from tensorflow.keras.models import load_model
import Config
from Data_loader import fetch_data
from Preprocessing import DataProcessor

# --- ConfigURATION ---
# On utilise l'API uniquement pour savoir QUELS sont les prochains matchs
NEXT_EVENTS_URL = f"https://www.thesportsdb.com/api/v1/json/{Config.API_KEY}/eventsnextleague.php?id={Config.LIGUE_1_ID}"

# Dictionnaire de correction de noms (API Name -> CSV Name)
# À compléter si tu vois des erreurs "Pas assez d'historique"
NAME_MAPPING = {
    "Paris Saint-Germain": "Paris SG",
    "St Etienne": "St_Etienne",
    "Olympique Marseille": "Marseille",
    "Olympique Lyonnais": "Lyon",
    # Ajoute d'autres corrections ici selon tes fichiers CSV
}

def get_upcoming_schedule():
    """Récupère la liste des prochains matchs via l'API"""
    print(f"📡 Récupération du calendrier via l'API...")
    try:
        response = requests.get(NEXT_EVENTS_URL)
        data = response.json()
        if not data or 'events' not in data or data['events'] is None:
            print("⚠️ API : Aucun match prévu trouvé (Limitation clé gratuite ou fin de saison).")
            # Pour le test, tu peux décommenter la ligne ci-dessous pour forcer un match :
            # return [{'strHomeTeam': 'Paris SG', 'strAwayTeam': 'Marseille'}] 
            return []
        return data['events']
    except Exception as e:
        print(f"❌ Erreur connexion API : {e}")
        return []

def get_last_n_matches_from_csv(df, team_name_api, n, processor):
    """
    Cherche l'historique dans le CSV local au lieu de l'API.
    """
    # 1. Gestion des différences de noms (API vs CSV)
    team_name = NAME_MAPPING.get(team_name_api, team_name_api)
    
    # 2. Filtrage dans le DataFrame chargé
    # On cherche l'équipe soit à domicile, soit à l'extérieur
    team_matches = df[(df['home_team'] == team_name) | (df['away_team'] == team_name)].copy()
    
    # 3. Vérification quantité de données
    if len(team_matches) < n:
        print(f"⚠️  Attention: Nom '{team_name}' (API: {team_name_api}) non trouvé ou pas assez de matchs dans le CSV.")
        return None

    # 4. On prend les N derniers matchs
    # Le DataFrame df est supposé déjà trié par date par fetch_data()
    last_matches = team_matches.tail(n)
    
    formatted_sequence = []
    
    # 5. Calcul des Features (Exactement comme à l'entraînement)
    for _, row in last_matches.iterrows():
        feats, _ = processor.calculate_match_features(row, team_name)
        formatted_sequence.append(feats)
        
    return np.array(formatted_sequence, dtype=np.float32)

def main():
    # 1. Chargement des outils
    print("📂 Chargement du modèle, du scaler et des données CSV...")
    try:
        model = load_model(Config.MODEL_PATH)
        with open(Config.SCALER_PATH, 'rb') as f:
            scaler = pickle.load(f)
        
        # On charge TOUT l'historique CSV en mémoire une seule fois
        df_history = fetch_data()
        processor = DataProcessor()
        
    except Exception as e:
        print(f"❌ Erreur de chargement : {e}")
        return

    # 2. Récupération du calendrier API
    fixtures = get_upcoming_schedule()
    if not fixtures: return

    print(f"\n🔮 PRÉDICTIONS PROCHAINE JOURNÉE ({len(fixtures)} matchs)")
    print("=" * 60)

    for match in fixtures:
        home_name_api = match['strHomeTeam']
        away_name_api = match['strAwayTeam']
        
        # 3. Récupération de l'historique depuis le CSV (Plus fiable !)
        history = get_last_n_matches_from_csv(df_history, home_name_api, Config.SEQUENCE_LENGTH, processor)
        
        if history is None:
            continue
            
        # 4. Normalisation (Comme à l'entraînement)
        # Reshape 2D -> Scaler -> Reshape 3D
        seq_2d = history.reshape(Config.SEQUENCE_LENGTH, Config.N_FEATURES)
        seq_scaled = scaler.transform(seq_2d)
        seq_final = seq_scaled.reshape(1, Config.SEQUENCE_LENGTH, Config.N_FEATURES)
        
        # 5. Prédiction
        pred = model.predict(seq_final, verbose=0)[0]
        
        # 6. Affichage
        win_prob = pred[2] * 100
        draw_prob = pred[1] * 100
        loss_prob = pred[0] * 100 # Défaite à domicile = Victoire extérieur
        
        winner_idx = np.argmax(pred)
        verdicts = [f"Victoire {away_name_api}", "Match Nul", f"Victoire {home_name_api}"]
        
        # Couleurs pour la console
        color = "\033[92m" if winner_idx == 2 else "\033[91m"
        reset = "\033[0m"

        print(f"⚽ {home_name_api} vs {away_name_api}")
        print(f"   📊 Forme (basée sur les 10 derniers matchs CSV) :")
        print(f"      {home_name_api} : {win_prob:.1f}%")
        print(f"      Nul          : {draw_prob:.1f}%")
        print(f"      {away_name_api} : {loss_prob:.1f}%")
        print(f"   🏆 Verdict IA : {color}{verdicts[winner_idx]}{reset}")
        print("-" * 60)

if __name__ == "__main__":
    main()