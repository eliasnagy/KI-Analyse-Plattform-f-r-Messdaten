import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader


#WARNUNG: NICHT AUF LAPTOP AUSFÜHREN lol


class FraesenDataset(Dataset):
    def __init__(self, sensor_ordner, wear_datei, fenster_groesse=1024, schritt_weite=512):
        """
        Liest alle Sensordaten und Verschleißwerte ein und erstellt Sliding Windows.
        """
        self.fenster_groesse = fenster_groesse
        self.daten_fenster = []
        self.labels = []
        
        # Lade die Zieldaten (Verschleiß)
        # Angenommen die CSV hat Spalten wie: 'datei_name', 'verschleiss_wert'
        wear_df = pd.read_csv(wear_datei).head(3)
        
        # Iteriere über jede Fräsung (jeden Eintrag in der wear.csv)
        for index, row in wear_df.iterrows():
            
            # 1. Dateinamen sauber zusammenbauen (c_1_001.csv etc.)
            datei_nummer = index + 1 
            datei_name = f"c_1_{int(datei_nummer):03d}.csv" 
            datei_pfad = os.path.join(sensor_ordner, datei_name)
            
            # 2. HIER HAT DIE VARIABLE GEFEHLT: Verschleiß auslesen!
            # Wichtig: 'flute_1' durch deinen echten Spaltennamen ersetzen
            verschleiss = float(row['flute_1']) 
            
            # 3. Prüfen, ob die Datei existiert
            if not os.path.exists(datei_pfad):
                print(f"Warnung: Datei nicht gefunden -> {datei_pfad}")
                continue
                
            # 4. Sensordaten laden 
            sensor_daten = pd.read_csv(datei_pfad).values 
            
            # 5. Daten in Fenster schneiden und speichern
            for start_idx in range(0, len(sensor_daten) - fenster_groesse, schritt_weite):
                fenster = sensor_daten[start_idx : start_idx + fenster_groesse]
                self.daten_fenster.append(fenster)
                self.labels.append(verschleiss)  # Hier wird 'verschleiss' jetzt gefunden!
                
        # Konvertiere die gesammelten Listen in schnelle PyTorch Tensoren
        self.daten_fenster = torch.tensor(np.array(self.daten_fenster), dtype=torch.float32)
        self.labels = torch.tensor(np.array(self.labels), dtype=torch.float32).view(-1, 1)
        
        # WICHTIG für 1D-CNNs in PyTorch: 
        # Das Format muss (Anzahl_Fenster, Kanäle, Sequenzlänge) sein.
        # Aktuell ist es (Anzahl, 1024, 7). Wir drehen es zu (Anzahl, 7, 1024).
        self.daten_fenster = self.daten_fenster.permute(0, 2, 1)

    def __len__(self):
        # Sagt PyTorch, wie viele Datenpakete wir insgesamt haben
        return len(self.daten_fenster)

    def __getitem__(self, idx):
        # Gibt ein einzelnes Fenster und den dazugehörigen Verschleißwert zurück
        return self.daten_fenster[idx], self.labels[idx]

# --- So benutzt du die Klasse am Ende in deinem Trainings-Skript ---

# 1. Datensatz initialisieren
mein_datensatz = FraesenDataset(
    sensor_ordner='./daten/c1', 
    wear_datei='./daten/c1_wear.csv',
    fenster_groesse=1024, # Das Netz sieht 1024 Messpunkte gleichzeitig
    schritt_weite=512     # Das nächste Fenster überlappt zur Hälfte
)

# 2. Dataloader erstellen
# Der Dataloader mischt die Daten und bündelt sie für die Grafikkarte in "Batches"
train_loader = DataLoader(
    mein_datensatz, 
    batch_size=32,  # 32 Fenster werden gleichzeitig berechnet
    shuffle=True    # Wichtig beim Training, damit das Netz nicht auswendig lernt
)

# 3. Test: Schauen wir uns ein Paket (Batch) an, das zur GPU geschickt wird
for sensordaten, verschleiss in train_loader:
    print(f"Sensordaten Form: {sensordaten.shape}") 
    print(f"Verschleiss Form: {verschleiss.shape}")
    break # Wir brechen nach dem ersten Paket ab