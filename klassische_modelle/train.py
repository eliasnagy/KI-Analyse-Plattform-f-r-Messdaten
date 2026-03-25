import os
import argparse
import numpy as np
import joblib
from datetime import datetime
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from config import Config
from data_processing import DatasetBuilder
from models import RandomForestModel, MLPModel


class Trainer:
    def __init__(self, args):
        self.args = args

    def run(self):
        BASE_DIR = self.args.base_dir
        
        # Nutze Model aus CLI oder fallback auf .env
        model_type = self.args.model or "MLP"
        
        # Input Ordner von CLI oder .env
        if self.args.input_folders:
            # CLI: komma-getrennte Liste
            input_folders = [os.path.join(BASE_DIR, f.strip()) for f in self.args.input_folders.split(',')]
        else:
            # Aus .env
            input_folders = [os.path.join(BASE_DIR, f) for f in Config.INPUT_FOLDERS]
        
        Config.print_config()
        print(f"📁 Lade Trainingsdaten aus {len(input_folders)} Ordner(n)...\n")

        # Lade Dataset aus allen Ordnern
        X_all, y_all = DatasetBuilder.build_or_load_dataset(input_folders, save_name="combined_features")

        # Train/Test Split
        X_train, X_test, y_train, y_test = train_test_split(
            X_all, y_all, test_size=Config.TEST_SPLIT_RATIO, random_state=Config.RANDOM_STATE
        )

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        print(f"\n--- Starte Training mit Modell: {model_type} ---")

        if model_type == "RandomForest":
            model = RandomForestModel(n_estimators=Config.RF_N_ESTIMATORS, max_depth=Config.RF_MAX_DEPTH)
            model.fit(X_train_scaled, y_train)
            train_predictions = model.predict(X_train_scaled)
            test_predictions = model.predict(X_test_scaled)

        elif model_type == "MLP":
            hidden = tuple(int(x) for x in Config.MLP_HIDDEN_LAYERS.split(",") if x.strip() != "")
            model = MLPModel(hidden_layer_sizes=hidden, activation=Config.MLP_ACTIVATION, solver=Config.MLP_SOLVER, max_iter=Config.MLP_MAX_ITER)
            model.fit(X_train_scaled, y_train)
            train_predictions = model.predict(X_train_scaled)
            test_predictions = model.predict(X_test_scaled)

        else:
            raise ValueError(f"Unbekanntes Modell: {model_type}")

        # ===== WICHTIG: Train & Test Metriken vergleichen =====
        train_mae = mean_absolute_error(y_train, train_predictions)
        test_mae = mean_absolute_error(y_test, test_predictions)
        train_mse = mean_squared_error(y_train, train_predictions)
        test_mse = mean_squared_error(y_test, test_predictions)
        train_rmse = np.sqrt(train_mse)
        test_rmse = np.sqrt(test_mse)
        train_mape = mean_absolute_percentage_error(y_train, train_predictions)
        test_mape = mean_absolute_percentage_error(y_test, test_predictions)
        train_r2 = r2_score(y_train, train_predictions)
        test_r2 = r2_score(y_test, test_predictions)

        print("-" * 60)
        print("TRAININGS vs. TEST PERFORMANCE:")
        print("-" * 60)
        print(f"{'Metrik':<15} {'Train':<15} {'Test':<15} {'Differenz':<15}")
        print("-" * 60)
        print(f"{'MAE':<15} {train_mae:<15.4f} {test_mae:<15.4f} {abs(test_mae - train_mae):<15.4f}")
        print(f"{'MSE':<15} {train_mse:<15.4f} {test_mse:<15.4f} {abs(test_mse - train_mse):<15.4f}")
        print(f"{'RMSE':<15} {train_rmse:<15.4f} {test_rmse:<15.4f} {abs(test_rmse - train_rmse):<15.4f}")
        print(f"{'MAPE':<15} {train_mape:<15.4f} {test_mape:<15.4f} {abs(test_mape - train_mape):<15.4f}")
        print(f"{'R² Score':<15} {train_r2:<15.4f} {test_r2:<15.4f} {abs(test_r2 - train_r2):<15.4f}")
        print("-" * 60)

        # ===== OVERFITTING ANALYSE =====
        r2_gap = train_r2 - test_r2  # Positive Werte = Overfitting
        
        print("\n OVERFITTING ANALYSE:")
        print(f"   R² Score Gap (Train - Test): {r2_gap:.4f}")
        
        if r2_gap > 0.15:  # Großer Gap deutet auf Overfitting hin
            print("   ⚠️  WARNUNG: STARKES OVERFITTING ERKANNT!")
            print(f"   Train-R² ({train_r2:.4f}) ist deutlich höher als Test-R² ({test_r2:.4f})")
            print("   → Empfehlung: Modell komplexität reduzieren oder mehr Trainings-Daten")
        elif r2_gap > 0.05:
            print("   ⚠️  HINWEIS: Leichtes Overfitting erkannt")
            print(f"   Gap: {r2_gap:.4f} (etwas höher als ideal)")
        else:
            print("   ✓ Gut: Kein signifikantes Overfitting erkannt!")
            print(f"   Train und Test Performance sind ähnlich (Gap: {r2_gap:.4f})")

            
        """
        train_errors = train_predictions - y_train
        test_errors = test_predictions - y_test
        
        print("\nALLE FEHLERWERTE - TRAININGSDATEN:")
        print("-" * 90)
        print(f"{'Idx':<8}{'Vorhersage':<14}{'Real':<14}{'Fehler (signiert)':<22}{'Abs. Fehler':<14}")
        print("-" * 90)
        for i in range(len(train_predictions)):
            print(
                f"{i+1:<8}{train_predictions[i]:<14.4f}{y_train[i]:<14.4f}"
                f"{train_errors[i]:<22.4f}{abs(train_errors[i]):<14.4f}"
            )

        print("\nALLE FEHLERWERTE - TESTDATEN:")
        print("-" * 90)
        print(f"{'Idx':<8}{'Vorhersage':<14}{'Real':<14}{'Fehler (signiert)':<22}{'Abs. Fehler':<14}")
        print("-" * 90)
        for i in range(len(test_predictions)):
            print(
                f"{i+1:<8}{test_predictions[i]:<14.4f}{y_test[i]:<14.4f}"
                f"{test_errors[i]:<22.4f}{abs(test_errors[i]):<14.4f}"
            )
        """

        # Modell speichern
        os.makedirs(Config.OUTPUT_FILES_FOLDER, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_filename = f"{Config.OUTPUT_FILES_FOLDER}/model_{model_type}_{timestamp}.pkl"
        scaler_filename = f"{Config.OUTPUT_FILES_FOLDER}/scaler_{model_type}_{timestamp}.pkl"
        
        joblib.dump(model.model, model_filename)
        joblib.dump(scaler, scaler_filename)
        
        print(f"\n✓ Modell gespeichert: {model_filename}")
        print(f"✓ Scaler gespeichert: {scaler_filename}")


def parse_args():
    parser = argparse.ArgumentParser(description="Trainiere ML-Modell für Werkzeugverschleiß-Vorhersage")
    parser.add_argument("--base-dir", default="./trainings_daten", help="Basisverzeichnis der Trainingsdaten (Standard: aus .env)")
    parser.add_argument("--model", choices=["RandomForest", "MLP"], default=None, help="Modell-Typ (Standard: aus .env)")
    parser.add_argument(
        "--input-folders", 
        default=None, 
        help="Komma-getrennte Input-Ordner (z.B. 'data_files/c1,data_files/c4'). Standard: aus .env"
    )
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    trainer = Trainer(args)
    trainer.run()