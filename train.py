import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

COLUMN_NAMES = ['Force_X', 'Force_Y', 'Force_Z', 'Vibration_X', 'Vibration_Y', 'Vibration_Z', 'AE_RMS']

def extract_features(file_path):
    # Lese CSV ohne Header
    df = pd.read_csv(file_path, header=None, names=COLUMN_NAMES)
    
    features = []
    for col in COLUMN_NAMES:
        data = df[col].values
        features.append(np.mean(data))
        features.append(np.std(data))
        features.append(np.max(data))
        features.append(np.min(data))
        rms = np.sqrt(np.mean(data**2))
        features.append(rms)
        
    return features

def build_or_load_dataset(cutter_folder, wear_file_path, save_name):
    """
    Lädt die Features, falls sie schon berechnet wurden.
    Ansonsten werden sie neu berechnet und als .npy Dateien gespeichert.
    """
    x_path = f"{save_name}_X.npy"
    y_path = f"{save_name}_y.npy"
    
    # Prüfen, ob die Arbeit schon mal gemacht haben
    if os.path.exists(x_path) and os.path.exists(y_path):
        print(f"Lade bereits extrahierte Features für {save_name}...")
        return np.load(x_path), np.load(y_path)
    
    print(f"Berechne Features neu für {cutter_folder}... (Das kann dauern)")
    
    # --- Anpassung für das korrekte Wear-File ---
    # CSV mit Header einlesen
    wear_data = pd.read_csv(wear_file_path)
    
    # Berechne den maximalen Verschleiß der 3 Schneiden pro Schnitt
    # Wir nehmen die Spalten flute_1, flute_2, flute_3 und suchen den Maximalwert pro Zeile (axis=1)
    wear_data['max_wear'] = wear_data[['flute_1', 'flute_2', 'flute_3']].max(axis=1)
    
    X = [] # Features
    y = [] # Zielvariable (maximaler Verschleiß pro Schnitt)
    
    csv_files = sorted([f for f in os.listdir(cutter_folder) if f.endswith('.csv')])
    
    for i, file in enumerate(csv_files):
        # Abbrechen, wenn wir keine Wear-Daten mehr für weitere Schnitte haben
        if i >= len(wear_data):
            break 
            
        file_path = os.path.join(cutter_folder, file)
        features = extract_features(file_path)
        
        X.append(features)
        # Nimm den zuvor berechneten max_wear Wert für diesen Schnitt
        y.append(wear_data['max_wear'].iloc[i]) 
        
    X_array = np.array(X)
    y_array = np.array(y)
    
    # Speichere die Arrays für den nächsten Durchlauf
    np.save(x_path, X_array)
    np.save(y_path, y_array)
    print(f"Features gespeichert unter {x_path} und {y_path}")
    
    return X_array, y_array

# ==========================================
# HAUPTPROGRAMM
# ==========================================

if __name__ == "__main__":
    BASE_DIR = "./trainingsdata" # Pfad zu deinen Daten anpassen!
    
    # Trainingsdaten (Cutter 1)
    train_folder = os.path.join(BASE_DIR, "input_folder")
    train_wear = os.path.join(BASE_DIR, "wear_files/c1_wear.csv")
    
    # Testdaten (Cutter 4)
    test_folder = os.path.join(BASE_DIR, "input_folder")
    test_wear = os.path.join(BASE_DIR, "wear_files/c1_wear.csv")
    
    # Datensätze laden oder neu aufbauen (mit Caching)
    X_train, y_train = build_or_load_dataset(train_folder, train_wear, "c1_features")
    X_test, y_test = build_or_load_dataset(test_folder, test_wear, "c4_features")
    
    print(f"\nTrainingsdaten Form: X={X_train.shape}, y={y_train.shape}")
    
    # Modell trainieren
    print("Trainiere Random Forest Modell...")
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Modell testen
    predictions = model.predict(X_test)
    
    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    
    print("-" * 30)
    print(f"Mean Absolute Error (MAE): {mae:.2f} (Abweichung in 10^-3 mm)")
    print(f"Root Mean Squared Error (RMSE): {np.sqrt(mse):.2f}")
    print("-" * 30)
    
    print("Beispiel-Vorhersagen vs. Realität (erste 5 Schnitte):")
    for i in range(5):
        print(f"Schnitt {i+1} -> Vorhergesagt: {predictions[i]:.2f}, Tatsächlicher Max-Verschleiß: {y_test[i]:.2f}")