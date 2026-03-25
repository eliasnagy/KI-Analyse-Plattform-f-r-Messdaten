KI-Analyse-Plattform für Messdaten
==================================

Vorhersage des Werkzeugverschleißes aus Sensormessdaten mit Machine Learning (Random Forest & MLP).

**Features:**
- ✅ Automatische Wear-Datei-Erkennung
- ✅ **Multi-Folder Support**: Kombiniere Daten von mehreren Cuttern für robustere Modelle
- ✅ Zentrale Konfiguration via `.env`
- ✅ Detaillierte Fehleranalyse und Overfitting-Erkennung
- ✅ Feature-Caching für schnelle Iteration

Übersicht
---------
- **Training**: `train.py` trainiert ein Modell und speichert es
- **Evaluierung**: `evaluate.py` lädt ein gespeichertes Modell und wertet es detailliert aus
- **Datenverarbeitung**: `data_processing.py` extrahiert Features aus CSV-Dateien
- **Modelle**: `models.py` definiert Random Forest und MLP Regressor
- **Konfiguration**: `.env` Datei für zentrale Parameterverwaltung

Installation
------------

```bash
pip install -r requirements.txt
```

Konfiguration (`.env` Datei)
----------------------------

Die `.env` Datei kontrolliert alle Parameter zentral:

```ini
# Folder Struktur
BASE_TRAINING_DIR=./trainingsdata

# Input Ordner(n) - komma-getrennte Liste für MEHRERE Ordner
# Einzelner Ordner: data_files/c1
# Mehrere Ordner: data_files/c1,data_files/c4,data_files/c6
INPUT_FOLDERS=data_files/c1
WEAR_FILES_FOLDER=wear_files
NUMPY_FILES_FOLDER=numpy_files
OUTPUT_FILES_FOLDER=output_files
```

**Wichtig**: 
- Nach Änderungen an `.env` muss das Programm neu gestartet werden
- Die Wear-Datei wird automatisch basierend auf Input-Dateien erkannt
- Du kannst mehrere Input-Ordner angeben, um Trainingsdaten zusammenzuführen!

Datenlayout
-----------
```
trainingsdata/
├── data_files/              # CSV-Dateien der Messdaten pro Cutter (Auto-erkannt)
│   ├── c1/
│   │   ├── c_1_001.csv
│   │   ├── c_1_002.csv
│   │   ⋮
│   ├── c4/
│   │   ├── c_4_001.csv
│   │   ├── c_4_002.csv
│   │   ⋮
│   └── c6/
│       ├── c_6_001.csv
│       └── ...
└── wear_files/              # Verschleiß-Daten (werden automatisch erkannt!)
    ├── c1_wear.csv
    ├── c4_wear.csv
    └── c6_wear.csv

numpy_files/                # Cache für extrahierte Features (automatisch erstellt)
├── combined_features_X.npy
└── combined_features_y.npy

output_files/               # Trainierte Modelle und Scaler (automatisch erstellt)
├── model_MLP_20260304_153021.pkl
├── scaler_MLP_20260304_153021.pkl
├── model_RandomForest_20260304_143015.pkl
└── scaler_RandomForest_20260304_143015.pkl
```

**Wie Wear-Dateien automatisch erkannt werden:**
- Input-Dateien: `c_1_*.csv` → Wear-Datei: `c1_wear.csv`
- Input-Dateien: `c_4_*.csv` → Wear-Datei: `c4_wear.csv`
- Input-Dateien: `c_6_*.csv` → Wear-Datei: `c6_wear.csv`

Training
--------

**Standard-Aufruf mit .env Konfiguration:**
```bash
python train.py
```

**Training mit einzelnem anderem Cutter (nur c4):**
```bash
python train.py --input-folders data_files/c4
```

**Training mit MEHREREN Cuttern kombiniert (empfohlen!):**
```bash
# Kombiniere c1, c4 und c6 Daten für robusteres Modell
python train.py --input-folders data_files/c1,data_files/c4,data_files/c6
```

**Oder in .env konfigurieren:**
```ini
INPUT_FOLDERS=data_files/c1,data_files/c4,data_files/c6
```

**Mit spezifischem Modell:**
```bash
python train.py --model RandomForest --input-folders data_files/c1,data_files/c4
```

### CLI-Parameter für `train.py`

**Verfügbare Parameter:**
- `--base-dir`: Basisverzeichnis der Trainingsdaten (Standard: aus `.env`)
- `--model`: `MLP` oder `RandomForest` (Standard: aus `.env`)
- `--input-folders`: Komma-getrennte Input-Ordner (Standard: aus `.env`)
  - Beispiel: `data_files/c1`
  - Mehrere: `data_files/c1,data_files/c4,data_files/c6`

**Hyperparameter werden über `.env` gesteuert:**
```bash
# .env bearbeiten:
RF_N_ESTIMATORS=200
RF_MAX_DEPTH=15
MLP_MAX_ITER=5000
MLP_HIDDEN_LAYERS=128,64,32
# dann speichern und Programm neu starten
```

### Output des Trainings
- **Metriken**: MAE, RMSE, R² Score (Train vs. Test)
- **Overfitting-Warnung**: Automatische Erkennung
- **Modell**: `output_files/model_[TYPE]_[TIMESTAMP].pkl`
- **Scaler**: `output_files/scaler_[TYPE]_[TIMESTAMP].pkl`
- **Beispiel-Vorhersagen**: Erste 5 Test-Samples mit Fehleranalyse

Modell-Evaluierung
------------------

**Evaluiere das neueste trainierte Modell:**
```bash
python evaluate.py --model MLP
```

**Evaluiere mit spezifischen Input-Ordnern:**
```bash
# Evaluiere mit den gleichen Ordnern wie beim Training
python evaluate.py --model MLP --input-folders data_files/c1,data_files/c4,data_files/c6
```

**Evaluiere ein spezifisches Modell:**
```bash
python evaluate.py --model MLP --model-path output_files/model_MLP_20260304_153021.pkl --scaler-path output_files/scaler_MLP_20260304_153021.pkl
```

### CLI-Parameter für `evaluate.py`

- `--model`: `MLP` oder `RandomForest` (Standard: aus `.env`)
- `--input-folders`: Komma-getrennte Input-Ordner (Standard: aus `.env`)
- `--model-path`: Pfad zur Modell-Datei (optional, sonst neustes wird verwendet)
- `--scaler-path`: Pfad zur Scaler-Datei (erforderlich wenn --model-path angegeben)

### Output der Evaluierung
- **Detaillierte Metriken**: MAE, RMSE, R², MAPE, Fehleranalyse
- **Top 10 beste Vorhersagen**: Mit einzelnen Fehlern
- **Top 10 schlechteste Vorhersagen**: Für Fehleranalyse

Produktivbetrieb / Inferenz
---------------------------

Nachdem ein Modell angelernt und die zugehörigen Scaler gespeichert wurden, kann das Tool im Produktivbetrieb verwendet werden, um für einzelne neue Messungen (ein "Schnitt" als CSV) Verschleiß vorherzusagen.

Beispielaufrufe:

```bash
# Verwende die jeweils neuesten Dateien in `output_files/`
python predict.py --input-file trainingsdata/data_files/c1/c_1_112.csv

# Modell und Scaler explizit angeben
python predict.py --input-file trainingsdata/data_files/c1/c_1_112.csv \
  --model-path output_files/model_MLP_20260315_163012.pkl \
  --scaler-path output_files/scaler_MLP_20260315_163012.pkl

# Vorhersage in Datei speichern
python predict.py --input-file trainingsdata/data_files/c1/c_1_112.csv --output pred.txt
```

Hinweise:
- `predict.py` lädt standardmäßig das neueste `model_*.pkl` und `scaler_*.pkl` aus `output_files/` (wenn keine Pfade angegeben werden).
- Die Eingabe-CSV muss das gleiche Format haben wie die Trainingsdaten (keine Header, 7 Spalten in der in `.env` konfigurierten Reihenfolge).
- `predict.py` gibt die geschätzte `max_wear` als einzelne Zahl aus.


Warum mehrere Folder kombinieren?
----------------------------------

Wenn du mehrere Input-Ordner kombinierst (z.B. `c1,c4,c6`), werden folgende Vorteile erreicht:

✅ **Robusteres Modell**: Mehr Trainingsdaten → bessere Generalisierung  
✅ **Bessere Vorhersagen**: Das Modell lernt über verschiedene Cutter-Typen  
✅ **Weniger Overfitting**: Größere Datenbasis reduziert Überanpassung  
✅ **Universal einsetzbar**: Ein Modell für mehrere Cutter-Konfigurationen  

**Beispiel - Vergleich:**
```bash
# Option 1: Nur c1 Daten (klein, kann schnell overfitting sein)
python train.py --input-folders data_files/c1

# Option 2: Kombiniert (groß, robuster) ✅ EMPFOHLEN
python train.py --input-folders data_files/c1,data_files/c4,data_files/c6
```
-----------------

```bash
# 1. Trainiere mit nur c1 Daten
python train.py --input-folders data_files/c1

# 2. Evaluiere mit c1 Daten
python evaluate.py --model MLP --input-folders data_files/c1

# 3. Trainiere mit KOMBINIERT Daten (besseres Modell!)
python train.py --input-folders data_files/c1,data_files/c4,data_files/c6

# 4. Evaluiere mit kombinierten Daten
python evaluate.py --model MLP --input-folders data_files/c1,data_files/c4,data_files/c6

# 5. Oder speichere kombinierte Ordner in .env
# INPUT_FOLDERS=data_files/c1,data_files/c4,data_files/c6
python train.py
python evaluate.py --model MLP
```

**Wichtig**: Die Wear-Datei wird automatisch basierend auf den Input-Dateien erkannt!

Technische Details
------------------

**Feature-Extraction:**
Aus jedem CSV werden 35 statistische Features extrahiert (pro Sensorkanal):
- Mean, Std-Dev, Max, Min, RMS (Root Mean Square)
- Für alle 7 Sensorkanäle → 35 Features total

**Train/Test Split:**
- Test-Ratio: Konfigurierbar via `.env` (Standard: 20%)
- Random Seed: Aus `.env` (Standard: 42, reproduzierbar)

**Skalierung:**
- MLP: StandardScaler (z-normalization)
- RandomForest: Keine Skalierung nötig

**Cache:**
Feature-Dateien werden in `numpy_files/` gecacht, um Durchläufe zu beschleunigen.
