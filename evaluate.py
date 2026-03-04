import os
import argparse
import numpy as np
import joblib
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from config import Config
from data_processing import DatasetBuilder


# ==========================================
# MODELL EVALUIERUNG
# ==========================================

class ModelEvaluator:
    def __init__(self, model_path, scaler_path, model_type):
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        self.model_type = model_type
        print(f"✓ Modell geladen: {model_path}")
        print(f"✓ Scaler geladen: {scaler_path}")

    def evaluate(self, X_test, y_test):
        """Führe umfassende Evaluierung durch."""
        
        # Skalierung wenn nötig
        if self.model_type == "MLP":
            X_test_scaled = self.scaler.transform(X_test)
            predictions = self.model.predict(X_test_scaled)
        else:
            predictions = self.model.predict(X_test)

        # Berechne Metriken
        mae = mean_absolute_error(y_test, predictions)
        mse = mean_squared_error(y_test, predictions)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, predictions)
        mape = mean_absolute_percentage_error(y_test, predictions)

        # Fehleranalyse
        errors = predictions - y_test
        error_mean = np.mean(errors)
        error_std = np.std(errors)
        error_max = np.max(np.abs(errors))

        # Ausgabe
        print("\n" + "="*60)
        print("MODELL EVALUIERUNGSERGEBNISSE")
        print("="*60)
        print(f"Anzahl Test-Samples: {len(y_test)}")
        print(f"\nMetriken:")
        print(f"  MAE  (Mean Absolute Error):         {mae:.4f} (10^-3 mm)")
        print(f"  RMSE (Root Mean Squared Error):     {rmse:.4f}")
        print(f"  R²   (Determinationskoeffizient):   {r2:.4f}")
        print(f"  MAPE (Mean Absolute % Error):       {mape:.4f}%")
        
        print(f"\nFehleranalyse:")
        print(f"  Fehler Durchschnitt:  {error_mean:.4f}")
        print(f"  Fehler Std-Abw.:      {error_std:.4f}")
        print(f"  Max absoluter Fehler: {error_max:.4f}")
        
        print(f"\nReale Werte - Spannbreite:")
        print(f"  Min: {np.min(y_test):.2f}, Max: {np.max(y_test):.2f}")
        print(f"  Mittel: {np.mean(y_test):.2f}")
        
        print("\n" + "="*60)
        print("TOP 10 BESTE VORHERSAGEN:")
        print("="*60)
        top_indices = np.argsort(np.abs(errors))[:10]
        for rank, idx in enumerate(top_indices, 1):
            print(f"{rank:2d}. Pred: {predictions[idx]:7.2f} | Real: {y_test[idx]:7.2f} | Fehler: {errors[idx]:+7.2f}")
        
        print("\n" + "="*60)
        print("TOP 10 SCHLECHTESTE VORHERSAGEN:")
        print("="*60)
        worst_indices = np.argsort(np.abs(errors))[-10:][::-1]
        for rank, idx in enumerate(worst_indices, 1):
            print(f"{rank:2d}. Pred: {predictions[idx]:7.2f} | Real: {y_test[idx]:7.2f} | Fehler: {errors[idx]:+7.2f}")

        return {
            'mae': mae,
            'mse': mse,
            'rmse': rmse,
            'r2': r2,
            'mape': mape,
            'predictions': predictions,
            'errors': errors
        }


def find_latest_model(model_type):
    """Finde das neueste Modell des angegebenen Typs."""
    output_dir = Config.OUTPUT_FILES_FOLDER
    if not os.path.exists(output_dir):
        raise FileNotFoundError(f"Verzeichnis '{output_dir}' nicht gefunden!")
    
    model_files = [f for f in os.listdir(output_dir) if f.startswith(f"model_{model_type}_")]
    if not model_files:
        raise FileNotFoundError(f"Keine Modelle vom Typ '{model_type}' gefunden!")
    
    # Sortiere nach Datum (neuste zuerst)
    model_files.sort(reverse=True)
    latest_model = model_files[0]
    timestamp = latest_model.replace(f"model_{model_type}_", "").replace(".pkl", "")
    
    model_path = os.path.join(output_dir, latest_model)
    scaler_path = os.path.join(output_dir, f"scaler_{model_type}_{timestamp}.pkl")
    
    print(f"Neuestes Modell gefunden: {latest_model}")
    
    return model_path, scaler_path


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluiere ein trainiertes ML-Modell")
    parser.add_argument("--model", choices=["RandomForest", "MLP"], default=None, help="Modell-Typ (Standard: aus .env)")
    parser.add_argument("--model-path", default=None, help="Pfad zum gespeicherten Modell (optional, neustes wird verwendet)")
    parser.add_argument("--scaler-path", default=None, help="Pfad zum Scaler (erforderlich wenn --model-path angegeben)")
    parser.add_argument("--base-dir", default="./trainingsdata", help="Basisverzeichnis der Trainingsdaten")
    parser.add_argument(
        "--input-folders",
        default=None,
        help="Komma-getrennte Input-Ordner (z.B. 'data_files/c1,data_files/c4'). Standard: aus .env"
    )
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    Config.print_config()
    
    # Nutze Model aus CLI oder fallback auf .env
    model_type = args.model or "MLP"
    
    # Modell-Pfade bestimmen
    if args.model_path is None:
        model_path, scaler_path = find_latest_model(model_type)
    else:
        model_path = args.model_path
        scaler_path = args.scaler_path
        if scaler_path is None:
            raise ValueError("--scaler-path ist erforderlich wenn --model-path angegeben wird!")
    
    # Daten laden
    BASE_DIR = args.base_dir
    
    # Input Ordner von CLI oder .env
    if args.input_folders:
        # CLI: komma-getrennte Liste
        input_folders = [os.path.join(BASE_DIR, f.strip()) for f in args.input_folders.split(',')]
    else:
        # Aus .env
        input_folders = [os.path.join(BASE_DIR, f) for f in Config.INPUT_FOLDERS]
    
    print(f"📁 Lade Daten aus {len(input_folders)} Ordner(n)...\n")
    X_all, y_all = DatasetBuilder.build_or_load_dataset(input_folders, save_name="combined_features")
    
    # Split (gleich wie beim Training)
    X_train, X_test, y_train, y_test = train_test_split(
        X_all, y_all, test_size=Config.TEST_SPLIT_RATIO, random_state=Config.RANDOM_STATE
    )
    
    # Evaluiere
    evaluator = ModelEvaluator(model_path, scaler_path, model_type)
    results = evaluator.evaluate(X_test, y_test)
