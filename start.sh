#!/bin/bash

# 1. Arbeitsverzeichnis auf den Projektordner fixieren
PROJEKT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
cd "$PROJEKT_DIR"

echo "------------------------------------------"
echo "MODELL AUSWAHL:"
echo "1) Random Forest"
echo "2) MLP"
echo "3) 1D-CNN"
echo "------------------------------------------"
read -p "Wähle eine Kategorie [1-3]: " hauptwahl

case $hauptwahl in
  1)
    echo -e "\n--- RANDOM FOREST ---"
    MODEL_TYPE="RandomForest" # Variable für das Argument setzen
    echo "a) Training"
    echo "b) Evaluation"
    read -p "Wähle Unteroption [a-b]: " unterwahl
    case $unterwahl in
      a)
        ./env_classic/bin/python3 ./klassische_modelle/train.py --model $MODEL_TYPE
        ;;
      b)
        ./env_classic/bin/python3 ./klassische_modelle/live.py --model $MODEL_TYPE
        ;;
      *) echo "Ungültige Unteroption.";;
    esac
    ;;

  2)
    echo -e "\n--- MULTILAYER PERCEPTRON ---"
    MODEL_TYPE="MLP" # Hier der String, den dein Python-Skript erwartet
    echo "a) Training"
    echo "b) Evaluation"
    read -p "Wähle Unteroption [a-b]: " unterwahl
    case $unterwahl in
      a)
        # Hier nutzt du ebenfalls das gleiche Training-Skript mit dem MLP-Parameter
        ./env_classic/bin/python3 ./klassische_modelle/train.py --model $MODEL_TYPE
        ;;
      b)
        ./env_classic/bin/python3 ./klassische_modelle/live.py --model $MODEL_TYPE
        ;;
      *) echo "Ungültige Unteroption.";;
    esac
    ;;

  3)
    echo -e "\n--- TORCH 1D-CNN ---"
    echo "a) Training"
    echo "b) GLive-Inferenz"
    read -p "Wähle Unteroption [a-b]: " unterwahl
    case $unterwahl in
      a)
        ./env_torch/bin/python3 ./torch/train.py
        ;;
      b)
        ./env_torch/bin/python3 ./torch/inference.py
        ;;
      *) echo "Ungültige Unteroption.";;
    esac
    ;;

  *)
    echo "Ungültige Hauptauswahl."
    exit 1
    ;;
esac