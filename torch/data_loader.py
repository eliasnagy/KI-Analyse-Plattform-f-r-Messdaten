import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
import glob

class FraesenDataset(Dataset):
    def __init__(self, sensor_folder, window_size=1024, step_size=512, global_mean=None, global_std=None, is_inference=False):
        self.sensor_folder = sensor_folder
        self.window_size = window_size
        self.step_size = step_size
        self.is_inference = is_inference
        
        folder_name = os.path.basename(os.path.normpath(sensor_folder))
        self.cache_file = os.path.join(sensor_folder, f"cache_{folder_name}_w{window_size}_s{step_size}.pt")
        
        self.raw_files = []
        self.index_map = []
        self.has_labels = False
        
        self._load_or_build_data()
        self._normalize(global_mean, global_std)

    def _detect_wear_file(self):
        # 1. Alle Sensor-CSV-Dateien finden, um die ID zu extrahieren
        csv_files = sorted([f for f in os.listdir(self.sensor_folder) if f.endswith('.csv') and 'wear' not in f])
        if not csv_files:
            return None
        
        # Extrahiere cutter_id (z.B. "c_1_001.csv" -> "c1")
        parts = csv_files[0].replace('.csv', '').split('_')
        if len(parts) < 2: 
            return None
        
        cutter_id = f"{parts[0]}{parts[1]}" 
        wear_filename = f"{cutter_id}_wear.csv"
        
        current_script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(current_script_dir)

        # 2. Pfad zum neuen Unterordner definieren
        wear_folder = os.path.join(project_root, "trainings_daten", "wear_files")
        path_wear = os.path.join(wear_folder, wear_filename)
        
        # 3. Prüfen, ob die Datei dort existiert
        if os.path.exists(path_wear):
            return path_wear
            
        return None
        

    def _load_or_build_data(self):
        if self.is_inference:
            print("Inferenz-Modus aktiv: Suche nicht nach Labels.")
            self.has_labels = False
            wear_file = None
        else:
            wear_file = self._detect_wear_file()
            self.has_labels = wear_file is not None

        if self.has_labels:
            wear_df = pd.read_csv(wear_file)
            wear_df['max_wear'] = wear_df[['flute_1', 'flute_2', 'flute_3']].max(axis=1)
            
            file_idx = 0
            for index, row in wear_df.iterrows():
                parts = os.path.basename(wear_file).replace('_wear.csv', '')
                file_name = f"{parts[0]}_{parts[1]}_{int(index + 1):03d}.csv"
                file_path = os.path.join(self.sensor_folder, file_name)
                
                if not os.path.exists(file_path): continue
                
                sensor_data = pd.read_csv(file_path).values.astype(np.float32)
                self.raw_files.append(sensor_data)
                
                wear_value = float(row['max_wear'])
                
                # --- THRESHOLDING LOGIK (Dynamisch) ---
                # Berechne die durchschnittliche Schwankung der gesamten Datei
                file_activity = np.std(sensor_data, axis=0).mean()
                # Setze den Schwellenwert auf z.B. 20% der Durchschnittsaktivität
                threshold = file_activity * 0.20 
                
                for start_idx in range(0, len(sensor_data) - self.window_size, self.step_size):
                    window = sensor_data[start_idx : start_idx + self.window_size]
                    
                    # Berechne die Aktivität nur für dieses spezifische Fenster
                    window_activity = np.std(window, axis=0).mean()
                    
                    # Ist das Fenster zu "ruhig"? -> Dann ist es ein Luftschnitt. Ignorieren!
                    if window_activity < threshold:
                        continue 
                        
                    self.index_map.append((file_idx, start_idx, wear_value))
                file_idx += 1
                
        else:
            search_pattern = os.path.join(self.sensor_folder, "**", "*.csv")
            all_files = sorted([f for f in glob.glob(search_pattern, recursive=True) if 'wear' not in f])
            file_idx = 0
            for file_path in all_files:
                sensor_data = pd.read_csv(file_path).values.astype(np.float32)
                self.raw_files.append(sensor_data)
                
                # Auch hier Thresholding anwenden, damit wir in der Inferenz keine Luft vorhersagen
                file_activity = np.std(sensor_data, axis=0).mean()
                threshold = file_activity * 0.20 
                
                for start_idx in range(0, len(sensor_data) - self.window_size, self.step_size):
                    window = sensor_data[start_idx : start_idx + self.window_size]
                    window_activity = np.std(window, axis=0).mean()
                    
                    if window_activity < threshold:
                        continue
                        
                    self.index_map.append((file_idx, start_idx, None))
                file_idx += 1

        torch.save({
            'raw_files': self.raw_files,
            'index_map': self.index_map,
            'has_labels': self.has_labels
        }, self.cache_file)
        print(f"Cache gespeichert ({len(self.index_map)} echte Schnitt-Fenster generiert).")


    def _normalize(self, global_mean, global_std):
        """Wendet Z-Score Normalisierung auf die geladenen Daten an."""
        if len(self.raw_files) == 0: return

        all_data_combined = np.vstack(self.raw_files)
        
        # Wenn kein Mean/Std übergeben wurde, berechnen wir sie selbst (passiert beim Trainings-Set)
        if global_mean is None or global_std is None:
            self.mean = all_data_combined.mean(axis=0)
            self.std = all_data_combined.std(axis=0)
            self.std[self.std == 0] = 1e-6  # Verhindert Division durch Null
        else:
            # Bei Validierung/Inferenz nutzen wir die Werte vom Trainings-Set
            self.mean = global_mean
            self.std = global_std

        # Anwenden auf alle Arrays in der Liste
        for i in range(len(self.raw_files)):
            self.raw_files[i] = (self.raw_files[i] - self.mean) / self.std



    def __len__(self):
        return len(self.index_map)



    def __getitem__(self, idx):
        file_idx, start_idx, label = self.index_map[idx]
        
        # Dynamisches Ausschneiden aus dem RAM
        window_data = self.raw_files[file_idx][start_idx : start_idx + self.window_size]
        window_tensor = torch.tensor(window_data.copy()).permute(1, 0)
        
        if self.has_labels:
            label_tensor = torch.tensor([label], dtype=torch.float32)
            return window_tensor, label_tensor
        else:
            return window_tensor