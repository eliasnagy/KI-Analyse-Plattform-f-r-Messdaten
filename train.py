import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import ConcatDataset, DataLoader
import pandas as pd
import numpy as np

from data_loader import FraesenDataset


# ==========================================
# 1. Das KI-Modell definieren (Das "Gehirn")
# ==========================================
class VerschleissCNN(nn.Module):
    def __init__(self):
        super(VerschleissCNN, self).__init__()
        
        # 1. Faltungsschicht
        self.conv1 = nn.Conv1d(in_channels=7, out_channels=32, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(32)        # Stabilisiert das Training
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(kernel_size=2)
        
        # 2. Faltungsschicht
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(64)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool1d(kernel_size=2)
        
        self.flatten = nn.Flatten()
        
        # 3. Lineare Schichten
        self.fc1 = nn.Linear(64 * 256, 128)
        self.dropout = nn.Dropout(p=0.5)     # "Vergisst" 50% der Neuronen zufällig
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        x = self.pool1(self.relu1(self.bn1(self.conv1(x))))
        x = self.pool2(self.relu2(self.bn2(self.conv2(x))))
        x = self.flatten(x)
        x = self.dropout(self.relu3(self.fc1(x)))
        x = self.fc2(x)
        return x


# ==========================================
# 2. Vorbereitung für das Training
# ==========================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training läuft auf: {device}")

# --- TRAININGS-DATEN --- (c1, c4)
train_c1 = FraesenDataset('./daten/c1', './daten/c1_wear.csv', fenster_groesse=1024, schritt_weite=1024)
train_c4 = FraesenDataset('./daten/c4', './daten/c4_wear.csv', fenster_groesse=1024, schritt_weite=1024)

datensatz_train = ConcatDataset([train_c1, train_c4])
train_loader = DataLoader(datensatz_train, batch_size=256, shuffle=True, num_workers=4, pin_memory=True)

# --- VALIDIERUNGS-DATEN --- (c6)
datensatz_val = FraesenDataset('./daten/c6', './daten/c6_wear.csv', fenster_groesse=1024)
val_loader = DataLoader(datensatz_val, batch_size=256, shuffle=False, num_workers=4, pin_memory=True)

modell = VerschleissCNN().to(device)
fehler_funktion = nn.MSELoss()
optimizer = optim.Adam(modell.parameters(), lr=0.001, weight_decay=1e-4)


# ==========================================
# 3. Die Trainings-Schleife (Training Loop)
# ==========================================

epochen = 50                       # Sicherheitsnetz – Early Stopping greift meist viel früher
beste_val_fehler = float('inf')
geduld = 10                         # Stoppt nach 10 Epochen ohne Verbesserung
geduld_zaehler = 0

for epoche in range(epochen):
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
    print(f"Epoche {epoche+1}/{epochen} | Train-Fehler: {durchschnitt_train:.4f} | Val-Fehler: {durchschnitt_val:.4f}")

    # ==============================
    # EARLY STOPPING
    # ==============================
    if durchschnitt_val < beste_val_fehler:
        beste_val_fehler = durchschnitt_val
        torch.save(modell.state_dict(), "bestes_modell.pth")
        geduld_zaehler = 0
        print(f"  --> Neues bestes Modell gespeichert! Val-Fehler: {beste_val_fehler:.4f}")
    else:
        geduld_zaehler += 1
        if geduld_zaehler >= geduld:
            print(f"Early Stopping nach Epoche {epoche+1}!")
            break

print("Training beendet!")


# ==========================================
# 4. Vorhersagen als CSV speichern
# ==========================================

# Bestes Modell laden (nicht das letzte!)
modell.load_state_dict(torch.load("bestes_modell.pth"))
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