import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
import glob

class FraesenDataset(Dataset):
    def __init__(self, sensor_ordner, fenster_groesse=1024, schritt_weite=512, globaler_mean=None, globale_std=None):
        self.sensor_ordner = sensor_ordner
        self.fenster_groesse = fenster_groesse
        self.schritt_weite = schritt_weite
        
        # Eindeutiger Name für die Cache-Datei im jeweiligen Ordner
        ordner_name = os.path.basename(os.path.normpath(sensor_ordner))
        self.cache_datei = os.path.join(sensor_ordner, f"cache_{ordner_name}_w{fenster_groesse}_s{schritt_weite}.pt")
        
        self.rohe_dateien = []
        self.index_map = []
        self.hat_labels = False
        
        # 1. Daten laden (entweder blitzschnell aus dem Cache oder frisch aus CSVs)
        self._daten_laden_oder_bauen()
        
        # 2. Normalisierung anwenden (Z-Score)
        self._normalisieren(globaler_mean, globale_std)

    def _detect_wear_file(self):
        """Sucht automatisch die passende Wear-Datei im übergeordneten Verzeichnis oder im selben Ordner."""
        csv_files = sorted([f for f in os.listdir(self.sensor_ordner) if f.endswith('.csv') and 'wear' not in f])
        if not csv_files:
            return None
        
        # Beispiel: "c_1_001.csv" -> ["c", "1", "001"] -> cutter_id = "c1"
        parts = csv_files[0].replace('.csv', '').split('_')
        if len(parts) < 2: return None
        
        cutter_id = f"{parts[0]}{parts[1]}" 
        wear_filename = f"{cutter_id}_wear.csv"
        
        # Suche im aktuellen Ordner oder im übergeordneten Ordner
        pfad_aktuell = os.path.join(self.sensor_ordner, wear_filename)
        pfad_parent = os.path.join(os.path.dirname(self.sensor_ordner), wear_filename)
        
        if os.path.exists(pfad_aktuell): return pfad_aktuell
        if os.path.exists(pfad_parent): return pfad_parent
        return None

    def _daten_laden_oder_bauen(self):
        """Lädt die Cache-Datei oder verarbeitet die CSVs und erstellt einen Cache."""
        if os.path.exists(self.cache_datei):
            print(f"Lade blitzschnell aus Cache: {self.cache_datei}")
            cache = torch.load(self.cache_datei)
            self.rohe_dateien = cache['rohe_dateien']
            self.index_map = cache['index_map']
            self.hat_labels = cache['hat_labels']
            return

        print(f"Erstelle neuen Cache für Ordner: {self.sensor_ordner} ...")
        wear_datei = self._detect_wear_file()
        self.hat_labels = wear_datei is not None

        if self.hat_labels:
            print(f"  ✓ Wear-Datei gefunden: {os.path.basename(wear_datei)}")
            wear_df = pd.read_csv(wear_datei)
            # DAS GOLD-NUGGET: Wir nehmen den maximalen Verschleiß aller 3 Schneiden!
            wear_df['max_wear'] = wear_df[['flute_1', 'flute_2', 'flute_3']].max(axis=1)
            
            datei_idx = 0
            for index, row in wear_df.iterrows():
                # Baut den Dateinamen passend zur ID (z.B. c_1_001.csv)
                parts = os.path.basename(wear_datei).replace('_wear.csv', '')
                datei_name = f"{parts[0]}_{parts[1]}_{int(index + 1):03d}.csv"
                datei_pfad = os.path.join(self.sensor_ordner, datei_name)
                
                if not os.path.exists(datei_pfad): continue
                
                sensor_daten = pd.read_csv(datei_pfad).values.astype(np.float32)
                self.rohe_dateien.append(sensor_daten)
                
                verschleiss = float(row['max_wear'])
                
                for start_idx in range(0, len(sensor_daten) - self.fenster_groesse, self.schritt_weite):
                    self.index_map.append((datei_idx, start_idx, verschleiss))
                datei_idx += 1
        else:
            print("  ℹ Keine Wear-Datei gefunden. Lade nur Sensordaten (Inferenz-Modus).")
            alle_dateien = sorted([f for f in glob.glob(os.path.join(self.sensor_ordner, "*.csv")) if 'wear' not in f])
            datei_idx = 0
            for datei_pfad in alle_dateien:
                sensor_daten = pd.read_csv(datei_pfad).values.astype(np.float32)
                self.rohe_dateien.append(sensor_daten)
                
                for start_idx in range(0, len(sensor_daten) - self.fenster_groesse, self.schritt_weite):
                    self.index_map.append((datei_idx, start_idx, None))
                datei_idx += 1

        # Cache auf der Festplatte/SD-Karte speichern
        torch.save({
            'rohe_dateien': self.rohe_dateien,
            'index_map': self.index_map,
            'hat_labels': self.hat_labels
        }, self.cache_datei)
        print(f"  ✓ Cache gespeichert ({len(self.index_map)} Fenster generiert).")

    def _normalisieren(self, globaler_mean, globale_std):
        """Wendet Z-Score Normalisierung auf die geladenen Daten an."""
        if len(self.rohe_dateien) == 0: return

        alle_daten_kombiniert = np.vstack(self.rohe_dateien)
        
        # Wenn kein Mean/Std übergeben wurde, berechnen wir sie selbst (passiert beim Trainings-Set)
        if globaler_mean is None or globale_std is None:
            self.mean = alle_daten_kombiniert.mean(axis=0)
            self.std = alle_daten_kombiniert.std(axis=0)
            self.std[self.std == 0] = 1e-6  # Verhindert Division durch Null
        else:
            # Bei Validierung/Inferenz nutzen wir die Werte vom Trainings-Set
            self.mean = globaler_mean
            self.std = globale_std

        # Anwenden auf alle Arrays in der Liste
        for i in range(len(self.rohe_dateien)):
            self.rohe_dateien[i] = (self.rohe_dateien[i] - self.mean) / self.std

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        datei_idx, start_idx, label = self.index_map[idx]
        
        # Dynamisches Ausschneiden aus dem RAM
        fenster_daten = self.rohe_dateien[datei_idx][start_idx : start_idx + self.fenster_groesse]
        fenster_tensor = torch.tensor(fenster_daten.copy()).permute(1, 0)
        
        if self.hat_labels:
            label_tensor = torch.tensor([label], dtype=torch.float32)
            return fenster_tensor, label_tensor
        else:
            return fenster_tensor