import os
import pandas as pd
import numpy as np
from config import Config

COLUMN_NAMES = Config.COLUMN_NAMES


def detect_wear_file(cutter_folder):
    """Erkenne automatisch die richtige Wear-Datei basierend auf Input-Dateien.
    
    Beispiel: Wenn Input-Ordner c_1_001.csv, c_1_002.csv enthält,
    wird automatisch c1_wear.csv gesucht.
    """
    csv_files = sorted([f for f in os.listdir(cutter_folder) if f.endswith('.csv')])
    
    if not csv_files:
        raise FileNotFoundError(f"Keine CSV-Dateien in {cutter_folder} gefunden!")
    
    # Extrahiere Pattern aus erstem Dateinamen, z.B. "c_1" aus "c_1_001.csv"
    first_file = csv_files[0]
    parts = first_file.replace('.csv', '').split('_')
    
    if len(parts) < 2:
        raise ValueError(f"Unerwartetes CSV-Dateiformat: {first_file}")
    
    # Konstruiere Wear-Datei-Name, z.B. "c1_wear.csv" aus "c_1_001.csv"
    cutter_id = f"{parts[0]}{parts[1]}"  # z.B. "c_1" → "c1"
    wear_filename = f"{cutter_id}_wear.csv"
    
    wear_path = os.path.join(Config.BASE_TRAINING_DIR, Config.WEAR_FILES_FOLDER, wear_filename)
    
    if not os.path.exists(wear_path):
        raise FileNotFoundError(
            f"Wear-Datei nicht gefunden: {wear_path}\n"
            f"Erwartet basierend auf Input-Dateien: {wear_filename}"
        )
    
    return wear_path, cutter_id


def extract_features(file_path):
    """Extrahiere einfache statistische Features aus einer Messdatei (CSV ohne Header)."""
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


class DatasetBuilder:
    """Hilfsklasse, die Features lädt oder berechnet und als .npy speichert."""

    @staticmethod
    def build_or_load_dataset(cutter_folders, save_name="combined_features"):
        """Baue oder lade ein Dataset aus einem oder mehreren Ordnern.
        
        Args:
            cutter_folders: List von Ordner-Pfaden ODER einzelner Ordner-Pfad (string)
            save_name: Name für .npy Cache-Dateien
        """
        # Konvertiere single string zu list
        if isinstance(cutter_folders, str):
            cutter_folders = [cutter_folders]
        
        os.makedirs(Config.NUMPY_FILES_FOLDER, exist_ok=True)
        x_path = f"{Config.NUMPY_FILES_FOLDER}/{save_name}_X.npy"
        y_path = f"{Config.NUMPY_FILES_FOLDER}/{save_name}_y.npy"

        if os.path.exists(x_path) and os.path.exists(y_path):
            print(f"Lade bereits extrahierte Features für {save_name}...")
            return np.load(x_path), np.load(y_path)

        print(f"Berechne Features neu für {len(cutter_folders)} Ordner(n)...")
        
        X = []
        y = []
        
        # Verarbeite jeden Ordner
        for folder_idx, cutter_folder in enumerate(cutter_folders, 1):
            print(f"  [{folder_idx}/{len(cutter_folders)}] Verarbeite: {cutter_folder}")
            
            # Auto-detect Wear-Datei
            wear_file_path, cutter_id = detect_wear_file(cutter_folder)
            print(f"    ✓ Auto-detected Wear-Datei: {cutter_id}_wear.csv")
            
            wear_data = pd.read_csv(wear_file_path)
            wear_data['max_wear'] = wear_data[['flute_1', 'flute_2', 'flute_3']].max(axis=1)
            
            csv_files = sorted([f for f in os.listdir(cutter_folder) if f.endswith('.csv')])
            
            for i, file in enumerate(csv_files):
                if i >= len(wear_data):
                    break
                
                file_path = os.path.join(cutter_folder, file)
                features = extract_features(file_path)
                
                X.append(features)
                y.append(wear_data['max_wear'].iloc[i])
            
            print(f"    ✓ {len(csv_files)} Samples geladen")

        X_array = np.array(X)
        y_array = np.array(y)

        np.save(x_path, X_array)
        np.save(y_path, y_array)
        print(f"\n✓ Features gespeichert: {x_path}")
        print(f"  Total Samples: {len(X_array)}")

        return X_array, y_array
