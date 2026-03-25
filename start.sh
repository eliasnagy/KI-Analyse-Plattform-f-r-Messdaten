#!/bin/bash

# Ermittelt den Ordner, in dem dieses Shell-Skript liegt
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)

# Wechsle in diesen Ordner
cd "$SCRIPT_DIR"

echo "Welches Programm?"
echo "1) Torch (NumPy 1.x)"
echo "2) Neu (NumPy 2.x)"
read -p "Auswahl: " wahl

case $wahl in
  1)
    # Jetzt sucht Python relativ zu SCRIPT_DIR
    ./env_torch/bin/python3 ./torch/train.py
    ;;
  2)
    ./env_new/bin/python3 ./klassische_modell/train.py
    ;;
esac