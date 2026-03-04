import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import ConcatDataset

# Hier importierst du deine eigene Klasse aus dem anderen Skript
# (Angenommen, du hast das Dataloader-Skript 'daten_loader.py' genannt)
from data_loader import FraesenDataset, DataLoader 

# ==========================================
# 1. Das KI-Modell definieren (Das "Gehirn")
# ==========================================
class VerschleissCNN(nn.Module):
    def __init__(self):
        super(VerschleissCNN, self).__init__()
        
        # 1. Faltungsschicht (Sucht nach ersten Mustern in den 7 Sensoren)
        # in_channels=7, weil du 7 Sensor-Spalten hast!
        self.conv1 = nn.Conv1d(in_channels=7, out_channels=32, kernel_size=5, padding=2)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(kernel_size=2) # Halbiert die Länge von 1024 auf 512
        
        # 2. Faltungsschicht (Sucht nach tieferen Mustern)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, padding=2)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool1d(kernel_size=2) # Halbiert die Länge von 512 auf 256
        
        # Das Fenster "flach" machen, um es in ein normales neuronales Netz zu stecken
        self.flatten = nn.Flatten()
        
        # 3. Lineare Schichten (Entscheidungsfindung)
        # 64 Kanäle * 256 restliche Datenpunkte = 16384
        self.fc1 = nn.Linear(64 * 256, 128)
        self.relu3 = nn.ReLU()
        
        # Die allerletzte Schicht gibt genau 1 Zahl aus: Den Verschleiß!
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        # Hier wird definiert, wie die Daten durch das Netz fließen
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.flatten(x)
        x = self.relu3(self.fc1(x))
        x = self.fc2(x)
        return x

# ==========================================
# 2. Vorbereitung für das Training
# ==========================================

# --- TRAININGS-DATEN ---
# 1. Alle Trainings-Datensätze einzeln laden
train_c1 = FraesenDataset('./daten/c1', './daten/c1_wear.csv', fenster_groesse=1024, schritt_weite=5000)
train_c4 = FraesenDataset('./daten/c4', './daten/c4_wear.csv', fenster_groesse=1024, schritt_weite=5000)

datensatz_train = ConcatDataset([train_c1, train_c4])

# Hardware-Check: CPU (Laptop) oder GPU (Jetson Orin)?
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training läuft auf: {device}")

# Modell erstellen und auf die Hardware schieben
modell = VerschleissCNN().to(device)

# Wie messen wir den Fehler? MSE (Mean Squared Error) ist perfekt für Kommazahlen
fehler_funktion = nn.MSELoss()

# Der "Lern-Algorithmus" (Adam ist der Goldstandard)
optimizer = optim.Adam(modell.parameters(), lr=0.001)

# Dataloader holen (hier mit den Werten für deinen Laptop-Test)
datensatz = FraesenDataset('./daten/c1', './daten/c1_wear.csv', fenster_groesse=1024, schritt_weite=5000)
train_loader = DataLoader(datensatz_train, batch_size=32, shuffle=True)


# ==========================================
# 3. Die Trainings-Schleife (Training Loop)
# ==========================================

epochen = 50 # Wie oft schaut sich die KI den kompletten Datensatz an?

for epoche in range(epochen):
    modell.train()
    laufender_fehler = 0.0
    
    for batch_idx, (sensordaten, wahrer_verschleiss) in enumerate(train_loader):
        
        # 1. Daten auf die GPU/CPU schieben!
        sensordaten = sensordaten.to(device)
        wahrer_verschleiss = wahrer_verschleiss.to(device)
        
        # 2. Altes Wissen vom letzten Schritt löschen
        optimizer.zero_grad()
        
        # 3. Vorhersage machen (Forward Pass)
        vorhersage = modell(sensordaten)
        
        # 4. Fehler berechnen (Wie weit ist Vorhersage vom echten Verschleiß weg?)
        fehler = fehler_funktion(vorhersage, wahrer_verschleiss)
        
        # 5. Aus Fehlern lernen (Backward Pass & Gewichte anpassen)
        fehler.backward()
        optimizer.step()
        
        laufender_fehler += fehler.item()
        
    print(f"Epoche {epoche+1}/{epochen} | Durchschnittlicher Fehler: {laufender_fehler/len(train_loader):.4f}")

print("Training beendet!")