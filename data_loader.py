import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
import glob

class FraesenDataset(Dataset):
    def __init__(self, sensor_ordner, wear_datei=None, fenster_groesse=1024, schritt_weite=512):
        self.fenster_groesse = fenster_groesse
        self.hat_labels = wear_datei is not None
        
        # Hier speichern wir die rohen Sensordaten (eine Matrix pro Datei)
        self.rohe_dateien = []
        
        # Die "Landkarte": Speichert (datei_index, start_zeile, label) für jedes Fenster
        self.index_map = []
        
        if self.hat_labels:
            wear_df = pd.read_csv(wear_datei)
            datei_idx = 0
            
            for index, row in wear_df.iterrows():
                datei_name = f"c_{sensor_ordner[-1]}_{int(index + 1):03d}.csv"
                datei_pfad = os.path.join(sensor_ordner, datei_name)
                
                if not os.path.exists(datei_pfad): 
                    continue
                
                # Als float32 laden, um RAM zu halbieren!
                sensor_daten = pd.read_csv(datei_pfad).values.astype(np.float32)
                self.rohe_dateien.append(sensor_daten)
                
                verschleiss = float(row['flute_1'])
                
                # Nur die Start-Indizes merken, nicht die Daten kopieren!
                for start_idx in range(0, len(sensor_daten) - fenster_groesse, schritt_weite):
                    self.index_map.append((datei_idx, start_idx, verschleiss))
                
                datei_idx += 1

        else:
            alle_dateien = sorted(glob.glob(os.path.join(sensor_ordner, "*.csv")))
            datei_idx = 0
            
            for datei_pfad in alle_dateien:
                sensor_daten = pd.read_csv(datei_pfad).values.astype(np.float32)
                self.rohe_dateien.append(sensor_daten)
                
                for start_idx in range(0, len(sensor_daten) - fenster_groesse, schritt_weite):
                    self.index_map.append((datei_idx, start_idx, None))
                
                datei_idx += 1

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        # 1. Wo liegt das Fenster? Auf der Karte nachschauen
        datei_idx, start_idx, label = self.index_map[idx]
        
        # 2. Genau dieses Fenster dynamisch aus der rohen Matrix ausschneiden
        fenster_daten = self.rohe_dateien[datei_idx][start_idx : start_idx + self.fenster_groesse]
        
        # 3. In Tensor umwandeln und Dimensionen tauschen (für Conv1d: Kanäle nach vorne)
        # .copy() ist wichtig, damit der Tensor im Speicher sauber anliegt
        fenster_tensor = torch.tensor(fenster_daten.copy()).permute(1, 0) 
        
        if self.hat_labels:
            # Label als Shape [1] zurückgeben
            label_tensor = torch.tensor([label], dtype=torch.float32)
            return fenster_tensor, label_tensor
        else:
            return fenster_tensor