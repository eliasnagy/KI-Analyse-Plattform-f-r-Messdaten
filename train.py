import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import ConcatDataset, DataLoader
from dotenv import load_dotenv
import pandas as pd
import numpy as np

from data_loader import FraesenDataset

load_dotenv()

WINDOW_SIZE = int(os.getenv("WINDOW_SIZE", 1024))
STEP_SIZE = int(os.getenv("STEP_SIZE", 1024))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", 256))
LEARNING_RATE = float(os.getenv("LEARNING_RATE", 0.0005))
WEIGHT_DECAY = float(os.getenv("WEIGHT_DECAY", 1e-4))
EPOCHS = int(os.getenv("EPOCHS", 50))
PATIENCE = int(os.getenv("PATIENCE", 10))
NUM_WORKERS = int(os.getenv("NUM_WORKERS", 4))
MODEL_PATH = os.getenv("MODEL_PATH", "bestes_modell.pth")

# ==========================================
# 1. Das KI-Modell definieren (Das "Gehirn")
# ==========================================
class VerschleissCNN(nn.Module):
    def __init__(self):
        super(VerschleissCNN, self).__init__()
        
        # 1. Block
        self.conv1 = nn.Conv1d(in_channels=7, out_channels=32, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(32)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(kernel_size=2)
        
        # 2. Block
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(64)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool1d(kernel_size=2)
        
        # --- DER OVERFITTING KILLER ---
        # Komprimiert die 256 verbleibenden Zeitschritte auf exakt 1 Wert pro Kanal
        self.global_pool = nn.AdaptiveAvgPool1d(1) 
        
        self.flatten = nn.Flatten()
        
        # 3. Lineare Schichten (Jetzt winzig und robust!)
        # Wir kommen mit 64 Werten an und gehen auf 32. 
        self.fc1 = nn.Linear(64, 32)
        self.dropout = nn.Dropout(p=0.5)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(32, 1)

    def forward(self, x):
        x = self.pool1(self.relu1(self.bn1(self.conv1(x))))
        x = self.pool2(self.relu2(self.bn2(self.conv2(x))))
        x = self.global_pool(x)
        x = self.flatten(x)
        x = self.dropout(self.relu3(self.fc1(x)))
        x = self.fc2(x)
        return x


if __name__ == "__main__":
    # ==========================================
    # 2. Vorbereitung für das Training
    # ==========================================

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training läuft auf: {device}")

    # --- TRAININGS-DATEN --- (c1, c4)
    train_c1 = FraesenDataset('./trainings_daten/c1', WINDOW_SIZE=1024, STEP_SIZE=1024)
    train_c4 = FraesenDataset('./trainings_daten/c4', WINDOW_SIZE=1024, STEP_SIZE=1024)

    # Wir müssen die Normalisierungs-Werte (Mean/Std) beider Trainingssets kombinieren.
    # Da sie ähnlich sein sollten, reicht es für den Anfang, einfach die von c1 als Basis zu nehmen 
    train_mean = train_c1.mean
    train_std = train_c1.std

    datensatz_train = ConcatDataset([train_c1, train_c4])

    # Hier die neuen DataLoader-Einstellungen für den Jetson!
    train_loader = DataLoader(datensatz_train, BATCH_SIZE=256, shuffle=True, NUM_WORKERS=4, pin_memory=True)

    # --- VALIDIERUNGS-DATEN --- (c6)
    # WICHTIG: Wir übergeben die Normalisierungswerte aus dem Training!
    datensatz_val = FraesenDataset('./trainings_daten/c6', WINDOW_SIZE=1024, STEP_SIZE=1024, global_mean=train_mean, global_std=train_std)
    val_loader = DataLoader(datensatz_val, BATCH_SIZE=256, shuffle=False, NUM_WORKERS=4, pin_memory=True)


    modell = VerschleissCNN().to(device)
    fehler_funktion = nn.MSELoss()
    optimizer = optim.Adam(modell.parameters(), LEARNING_RATE=0.0005, WEIGHT_DECAY=1e-4)


    # ==========================================
    # 3. Die Trainings-Schleife (Training Loop)
    # ==========================================

    EPOCHS = 50                       # Sicherheitsnetz – Early Stopping greift meist viel früher
    best_val_error = float('inf')
    PATIENCE = 10                     # Stoppt nach 10 EPOCHS ohne Verbesserung
    patience_counter = 0

    for epoche in range(EPOCHS):
        modell.train()
        train_fehler_summe = 0.0

        # ==============================
        # TRAINING (c1, c4)
        # ==============================
        for batch_idx, (sensordaten, wahrer_verschleiss) in enumerate(train_loader):
            sensordaten = sensordaten.to(device)
            wahrer_verschleiss = wahrer_verschleiss.to(device)
            
            optimizer.zero_grad()
            vorhersage = modell(sensordaten)
            fehler = fehler_funktion(vorhersage, wahrer_verschleiss)
            fehler.backward()
            optimizer.step()
            
            train_fehler_summe += fehler.item()
        
        durchschnitt_train = train_fehler_summe / len(train_loader)

        # ==============================
        # VALIDIERUNG (c6)
        # ==============================
        modell.eval()
        val_fehler_summe = 0.0
        
        with torch.no_grad():
            for sensordaten_val, wahrer_verschleiss_val in val_loader:
                sensordaten_val, wahrer_verschleiss_val = sensordaten_val.to(device), wahrer_verschleiss_val.to(device)
                vorhersage_val = modell(sensordaten_val)
                fehler_val = fehler_funktion(vorhersage_val, wahrer_verschleiss_val)
                val_fehler_summe += fehler_val.item()
                
        durchschnitt_val = val_fehler_summe / len(val_loader)
        print(f"Epoche {epoche+1}/{EPOCHS} | Train-Fehler: {durchschnitt_train:.4f} | Val-Fehler: {durchschnitt_val:.4f}")

        # ==============================
        # EARLY STOPPING
        # ==============================
        if durchschnitt_val < best_val_error:
            best_val_error = durchschnitt_val
            
            torch.save({
                'modell_gewichte': modell.state_dict(),
                'train_mean': torch.tensor(train_mean),  # <-- In Tensor umwandeln
                'train_std': torch.tensor(train_std)     # <-- In Tensor umwandeln
            }, "bestes_modell.pth")
            
            patience_counter = 0
            print(f"  --> Neues bestes Modell gespeichert! Val-Fehler: {best_val_error:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"Early Stopping nach Epoche {epoche+1}!")
                break

    print("Training beendet!")


    # ==========================================
    # 4. Vorhersagen als CSV speichern
    # ==========================================

    # Modell laden
    checkpoint = torch.load(MODEL_PATH, weights_only=True)

    modell.load_state_dict(checkpoint['modell_gewichte'])
    train_mean = checkpoint['train_mean'].numpy()
    train_std = checkpoint['train_std'].numpy()

    modell.eval()

    alle_vorhersagen = []
    alle_echten_werte = []

    with torch.no_grad():
        for sensordaten_val, wahrer_verschleiss_val in val_loader:
            sensordaten_val = sensordaten_val.to(device)
            vorhersage_val = modell(sensordaten_val)
            
            # Von GPU zurück zu numpy
            alle_vorhersagen.extend(vorhersage_val.cpu().numpy().flatten().tolist())
            alle_echten_werte.extend(wahrer_verschleiss_val.numpy().flatten().tolist())

    # Absoluter Fehler pro Fenster
    abweichung = [abs(v - e) for v, e in zip(alle_vorhersagen, alle_echten_werte)]

    ergebnis_df = pd.DataFrame({
        'fenster_index':    range(len(alle_vorhersagen)),
        'echter_verschleiss':    alle_echten_werte,
        'vorhergesagter_verschleiss': alle_vorhersagen,
        'absoluter_fehler': abweichung
    })

    ergebnis_df.to_csv('vorhersagen_c6.csv', index=False)
    print(f"Vorhersagen gespeichert: vorhersagen_c6.csv ({len(ergebnis_df)} Fenster)")
    print(f"Durchschnittlicher Fehler: {np.mean(abweichung):.4f}")



    # export für TensorRT:
    """
    dummy_input = torch.randn(1, 7, 1024, device=device)
    torch.onnx.export(modell, dummy_input, "verschleiss_modell.onnx", input_names=['sensordaten'], output_names=['verschleiss'])
    """