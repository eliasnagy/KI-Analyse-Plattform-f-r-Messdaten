import torch
import os
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from data_loader import FraesenDataset
from train import VerschleissCNN
from dotenv import load_dotenv

load_dotenv()

MODELL_PFAD = os.getenv("MODEL_PATH", "bestes_modell.pth")
LIVE_DATEN_ORDNER = os.getenv("LIVE_DATA_DIR", "./live_daten/c2/")

WINDOW_SIZE = int(os.getenv("WINDOW_SIZE", 1024))
STEP_SIZE = int(os.getenv("STEP_SIZE", 1024))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", 128))
NUM_WORKERS = int(os.getenv("NUM_WORKERS", 4))
GRENZWERT = float(os.getenv("GRENZWERT_VERSCHLEISS", 100.0))

# Orin auf Höchstleistung prüfen
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Starte Live-Inferenz auf: {device}")

# --- 2. MODELL & NORMALISIERUNG LADEN ---
print("Lade Modell und Normalisierungs-Werte...")
checkpoint = torch.load(MODELL_PFAD, map_location=device)

# Modell initialisieren und Gewichte laden
modell = VerschleissCNN().to(device)
modell.load_state_dict(checkpoint['modell_gewichte'])
modell.eval() # WICHTIG: Schaltet Dropout und BatchNorm in den Vorhersage-Modus!

# Normalisierungswerte aus dem Training extrahieren
train_mean = checkpoint['train_mean'].cpu().numpy()
train_std = checkpoint['train_std'].cpu().numpy()

# --- 3. LIVE-DATEN LADEN ---
live_dataset = FraesenDataset(
    sensor_folder=LIVE_DATEN_ORDNER, 
    window_size=WINDOW_SIZE,      # <-- Hier Variable nutzen
    step_size=STEP_SIZE,          # <-- Hier Variable nutzen
    global_mean=train_mean,      
    global_std=train_std,
    is_inference=True
)

live_loader = DataLoader(
    live_dataset, 
    batch_size=BATCH_SIZE,        # <-- Hier Variable nutzen
    shuffle=False, 
    num_workers=NUM_WORKERS,      # <-- Hier Variable nutzen
    pin_memory=True
)

if len(live_dataset) == 0:
    print("Keine auswertbaren Daten gefunden!")
    exit()

# --- 4. VORHERSAGE MACHEN ---
print(f"Starte Auswertung von {len(live_dataset)} Zeitfenstern...")
alle_vorhersagen = []

# torch.no_grad() ist extrem wichtig! Es spart massig RAM und Rechenzeit, 
# da wir keine Gradienten fürs Training mehr berechnen müssen.
with torch.no_grad():
    for batch_idx, sensordaten in enumerate(live_loader):
        # Daten auf die Jetson GPU schieben
        sensordaten = sensordaten.to(device)
        
        # Vorhersage des CNNs
        vorhersage = modell(sensordaten)
        
        # Werte von der GPU zurück in eine normale Python-Liste holen
        alle_vorhersagen.extend(vorhersage.cpu().numpy().flatten().tolist())

# --- 5. ERGEBNISSE AUSWERTEN ---
durchschnittlicher_verschleiss = np.mean(alle_vorhersagen)
maximaler_verschleiss = np.max(alle_vorhersagen)

print("-" * 50)
print("ERGEBNIS DER LIVE-AUSWERTUNG:")
print("-" * 50)
print(f"Anzahl ausgewerteter Fenster: {len(alle_vorhersagen)}")
print(f"Durchschnittlich geschätzter Verschleiß: {durchschnittlicher_verschleiss:.4f}")
print(f"Maximal geschätzter Verschleiß:        {maximaler_verschleiss:.4f}")

if maximaler_verschleiss > GRENZWERT:
    print("\nALARM: Kritischer Werkzeugverschleiß erkannt! Maschine stoppen!")

# Ergebnisse als CSV speichern
ergebnis_df = pd.DataFrame({
    'fenster_index': range(len(alle_vorhersagen)),
    'vorhergesagter_verschleiss': alle_vorhersagen
})
ergebnis_df.to_csv('live_vorhersagen.csv', index=False)
print("\nDetaillierte Vorhersagen gespeichert in: live_vorhersagen.csv")