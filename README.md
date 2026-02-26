KI-Analyse-Plattform für Messdaten
=================================

Kurz: Trainingsskript zur Vorhersage des Werkzeugverschleißes aus Sensordaten.

Datenlayout
-----------
- Trainingsdaten: `trainingsdata/input_folder/` (CSV-Dateien pro Schnitt)
- Wear-Daten: `trainingsdata/wear_files/`

Wichtig: Das Skript speichert extrahierte Features als `c1_features_X.npy` / `c1_features_y.npy` bzw. `test_features_X.npy` / `test_features_y.npy` im aktuellen Verzeichnis.

Usage / Beispiele
------------------
Standard-Aufruf (verwendet MLP mit Standard-Parametern):

```bash
python train.py
```

MLP mit expliziten Parametern:

```bash
python train.py --model MLP --mlp-hidden-layers 128,64 --mlp-max-iter 500 --mlp-activation relu --mlp-solver adam
```

Random Forest Beispiel:

```bash
python train.py --model RandomForest --rf-n-estimators 200 --rf-max-depth 10
```

Wichtige CLI-Parameter und Standardwerte
---------------------------------------
- `--model`: `MLP` (Standard) oder `RandomForest`
- `--base-dir`: `./trainingsdata` (Pfad zu deinen Daten)
- RandomForest:
	- `--rf-n-estimators`: 100
	- `--rf-max-depth`: None (kein Limit)
- MLP:
	- `--mlp-hidden-layers`: "100,50"
	- `--mlp-max-iter`: 1000
	- `--mlp-activation`: `relu` (Optionen: `identity`, `logistic`, `tanh`, `relu`)
	- `--mlp-solver`: `adam` (Optionen: `lbfgs`, `sgd`, `adam`)

Ergebnis
--------
Das Skript gibt MAE/RMSE aus und einige Beispiel-Vorhersagen. Die extrahierten Feature-Dateien bleiben für schnellere Wiederholläufe erhalten.

Fragen oder Änderungswünsche?
----------------------------
Sollen wir zusätzlich eine `requirements.txt` erstellen oder Default-Parameter im Code weiter anpassen? 
