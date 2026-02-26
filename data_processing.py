import os
import pandas as pd
import numpy as np

COLUMN_NAMES = ['Force_X', 'Force_Y', 'Force_Z', 'Vibration_X', 'Vibration_Y', 'Vibration_Z', 'AE_RMS']


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
    def build_or_load_dataset(cutter_folder, wear_file_path, save_name):
        os.makedirs("numpy_files", exist_ok=True)
        x_path = f"numpy_files/{save_name}_X.npy"
        y_path = f"numpy_files/{save_name}_y.npy"

        if os.path.exists(x_path) and os.path.exists(y_path):
            print(f"Lade bereits extrahierte Features für {save_name}...")
            return np.load(x_path), np.load(y_path)

        print(f"Berechne Features neu für {cutter_folder}... (Das kann dauern)")

        wear_data = pd.read_csv(wear_file_path)
        wear_data['max_wear'] = wear_data[['flute_1', 'flute_2', 'flute_3']].max(axis=1)

        X = []
        y = []

        csv_files = sorted([f for f in os.listdir(cutter_folder) if f.endswith('.csv')])

        for i, file in enumerate(csv_files):
            if i >= len(wear_data):
                break

            file_path = os.path.join(cutter_folder, file)
            features = extract_features(file_path)

            X.append(features)
            y.append(wear_data['max_wear'].iloc[i])

        X_array = np.array(X)
        y_array = np.array(y)

        np.save(x_path, X_array)
        np.save(y_path, y_array)
        print(f"Features gespeichert unter {x_path} und {y_path}")

        return X_array, y_array
