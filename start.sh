#!/bin/bash

# Den absoluten Pfad zum Projekt-Ordner herausfinden
PROJEKT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
cd "$PROJEKT_DIR"

echo "Was soll gestartet werden?"
echo "1) Torch Skript (NumPy 1.x)"
echo "2) Klassisches Modell (NumPy 2.2.6)"
read -p "Auswahl: " wahl

case $wahl in
  1)
    echo "Starte Torch..."
    # Wir rufen Python auf, bleiben aber im Hauptordner stehen!
    ./env_torch/bin/python3 ./torch/train.py
    ;;
  2)
    echo "Starte Klassisches Modell..."
    ./env_classic/bin/python3 ./klassische_modelle/train.py
    ;;
  *)
    echo "Abbruch."
    ;;
esac