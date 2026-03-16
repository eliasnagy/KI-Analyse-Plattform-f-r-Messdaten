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
        self.is_inference = is_inference  # <-- NEU: Schalter merken
        
        folder_name = os.path.basename(os.path.normpath(sensor_folder))
        self.cache_file = os.path.join(sensor_folder, f"cache_{folder_name}_w{window_size}_s{step_size}.pt")
        
        self.raw_files = []
        self.index_map = []
        self.has_labels = False
        
        self._load_or_build_data()
        self._normalize(global_mean, global_std)

    def _detect_wear_file(self):
            """Sucht automatisch die passende Wear-Datei im übergeordneten Verzeichnis oder im selben Ordner."""
            csv_files = sorted([f for f in os.listdir(self.sensor_folder) if f.endswith('.csv') and 'wear' not in f])
            if not csv_files:
                return None
            
            # Beispiel: "c_1_001.csv" -> ["c", "1", "001"] -> cutter_id = "c1"
            parts = csv_files[0].replace('.csv', '').split('_')
            if len(parts) < 2: return None
            
            cutter_id = f"{parts[0]}{parts[1]}" 
            wear_filename = f"{cutter_id}_wear.csv"
            
            # Suche im aktuellen Ordner oder im übergeordneten Ordner
            path_current = os.path.join(self.sensor_folder, wear_filename)
            path_parent = os.path.join(os.path.dirname(self.sensor_folder), wear_filename)
            
            if os.path.exists(path_current): return path_current
            if os.path.exists(path_parent): return path_parent
            return None
    
    
    def _load_or_build_data(self):
        if os.path.exists(self.cache_file):
            try:
                # ... (Dein bisheriger Cache-Lade Code) ...
                return
            except Exception as e:
                os.remove(self.cache_file)

        print(f"Erstelle neuen Cache für Ordner: {self.sensor_folder} ...")
        
        # --- NEU: HIER WIRD DER SCHALTER BENUTZT ---
        if self.is_inference:
            print("  ℹ Inferenz-Modus aktiv: Suche nicht nach Labels.")
            self.has_labels = False
            wear_file = None
        else:
            wear_file = self._detect_wear_file()
            self.has_labels = wear_file is not None

        # ... (Ab hier geht dein bisheriger Code ganz normal mit if self.has_labels: weiter) ...
        if self.has_labels:
            print(f"  ✓ Wear-Datei gefunden: {os.path.basename(wear_file)}")
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
                
                for start_idx in range(0, len(sensor_data) - self.window_size, self.step_size):
                    self.index_map.append((file_idx, start_idx, wear_value))
                file_idx += 1
        else:
            print("  ℹ Keine Wear-Datei gefunden. Lade nur Sensordaten (Inferenz-Modus).")
            all_files = sorted([f for f in glob.glob(os.path.join(self.sensor_folder, "*.csv")) if 'wear' not in f])
            file_idx = 0
            for file_path in all_files:
                sensor_data = pd.read_csv(file_path).values.astype(np.float32)
                self.raw_files.append(sensor_data)
                
                for start_idx in range(0, len(sensor_data) - self.window_size, self.step_size):
                    self.index_map.append((file_idx, start_idx, None))
                file_idx += 1

        torch.save({
            'raw_files': self.raw_files,
            'index_map': self.index_map,
            'has_labels': self.has_labels
        }, self.cache_file)
        print(f"  ✓ Cache gespeichert ({len(self.index_map)} Fenster generiert).")



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