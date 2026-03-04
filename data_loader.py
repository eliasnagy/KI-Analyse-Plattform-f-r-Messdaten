import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
import glob


class FraesenDataset(Dataset):
    def __init__(self, sensor_ordner, wear_datei=None, fenster_groesse=1024, schritt_weite=512):
        self.fenster_groesse = fenster_groesse
        self.daten_fenster = []
        self.labels = []
        self.hat_labels = wear_datei is not None # Merken wir uns für später
        
        # --- FALL 1: WIR HABEN VERSCHLEISSDATEN (Training & Validierung) ---
        if self.hat_labels:
            wear_df = pd.read_csv(wear_datei)
            for index, row in wear_df.iterrows():
                datei_name = f"c_{sensor_ordner[-1]}_{int(index + 1):03d}.csv" # Passe das an deine Benennung an
                datei_pfad = os.path.join(sensor_ordner, datei_name)
                verschleiss = float(row['flute_1']) 
                
                if not os.path.exists(datei_pfad): continue
                sensor_daten = pd.read_csv(datei_pfad).values 
                
                for start_idx in range(0, len(sensor_daten) - fenster_groesse, schritt_weite):
                    self.daten_fenster.append(sensor_daten[start_idx : start_idx + fenster_groesse])
                    self.labels.append(verschleiss)
                    
            self.labels = torch.tensor(np.array(self.labels), dtype=torch.float32).view(-1, 1)

        # --- FALL 2: WIR HABEN KEINE VERSCHLEISSDATEN (Echter Einsatz / Produktion) ---
        else:
            # Suche einfach ALLE .csv Dateien im Sensor-Ordner
            alle_dateien = sorted(glob.glob(os.path.join(sensor_ordner, "*.csv")))
            for datei_pfad in alle_dateien:
                sensor_daten = pd.read_csv(datei_pfad).values
                for start_idx in range(0, len(sensor_daten) - fenster_groesse, schritt_weite):
                    self.daten_fenster.append(sensor_daten[start_idx : start_idx + fenster_groesse])

        # Sensordaten umwandeln (passiert in beiden Fällen)
        self.daten_fenster = torch.tensor(np.array(self.daten_fenster), dtype=torch.float32)
        self.daten_fenster = self.daten_fenster.permute(0, 2, 1)

    def __len__(self):
        return len(self.daten_fenster)

    def __getitem__(self, idx):
        # Wenn wir Labels haben, geben wir (Daten, Label) zurück. Sonst nur die Daten.
        if self.hat_labels:
            return self.daten_fenster[idx], self.labels[idx]
        else:
            return self.daten_fenster[idx]