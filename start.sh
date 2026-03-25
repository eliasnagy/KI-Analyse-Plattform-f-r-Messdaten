#!/bin/bash

echo "Welches Programm möchtest du starten?"
echo "1) Torch-Programm (NumPy 1.x)"
echo "2) Neues Programm (NumPy 2.2.6)"
read -p "Auswahl [1-2]: " wahl

case $wahl in
  1)
    echo "Starte Torch-Programm..."
    ./env_torch/bin/python3 torch/train.py
    ;;
  2)
    echo "Starte neues Programm..."
    ./env_new/bin/python3 klassische_modelle/train.py
    ;;
  *)
    echo "Ungültige Auswahl."
    ;;
esac