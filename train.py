import os
import argparse
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler

from data_processing import DatasetBuilder
from models import RandomForestModel, MLPModel


# ==========================================
# HAUPTPROGRAMM
# ==========================================

class Trainer:
    def __init__(self, args):
        self.args = args

    def run(self):
        BASE_DIR = self.args.base_dir

        train_folder = os.path.join(BASE_DIR, "input_folder")
        train_wear = os.path.join(BASE_DIR, "wear_files/c1_wear.csv")

        test_folder = os.path.join(BASE_DIR, "input_folder")
        test_wear = os.path.join(BASE_DIR, "wear_files/c1_wear.csv")

        X_train, y_train = DatasetBuilder.build_or_load_dataset(train_folder, train_wear, "c1_features")
        X_test, y_test = DatasetBuilder.build_or_load_dataset(test_folder, test_wear, "test_features")

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        print(f"\n--- Starte Training mit Modell: {self.args.model} ---")

        if self.args.model == "RandomForest":
            rf_max_depth = None if self.args.rf_max_depth in (None, 'None') else int(self.args.rf_max_depth)
            model = RandomForestModel(n_estimators=self.args.rf_n_estimators, max_depth=rf_max_depth)
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)

        elif self.args.model == "MLP":
            hidden = tuple(int(x) for x in self.args.mlp_hidden_layers.split(",") if x.strip() != "")
            model = MLPModel(hidden_layer_sizes=hidden, activation=self.args.mlp_activation, solver=self.args.mlp_solver, max_iter=self.args.mlp_max_iter)
            model.fit(X_train_scaled, y_train)
            predictions = model.predict(X_test_scaled)

        else:
            raise ValueError("Unbekanntes Modell ausgewählt!")

        mae = mean_absolute_error(y_test, predictions)
        mse = mean_squared_error(y_test, predictions)

        print("-" * 30)
        print(f"Mean Absolute Error (MAE): {mae:.2f} (Abweichung in 10^-3 mm)")
        print(f"Root Mean Squared Error (RMSE): {np.sqrt(mse):.2f}")
        print("-" * 30)

        print("Beispiel-Vorhersagen vs. Realität (erste 5 Schnitte):")
        for i in range(min(5, len(predictions))):
            print(f"Schnitt {i+1} -> Vorhergesagt: {predictions[i]:.2f}, Tatsächlich: {y_test[i]:.2f}")


def parse_args():
    parser = argparse.ArgumentParser(description="Trainingsskript mit konfigurierbaren Modellen und Hyperparametern")
    parser.add_argument("--base-dir", default="./trainingsdata", help="Basisverzeichnis der Trainingsdaten")
    parser.add_argument("--model", choices=["RandomForest", "MLP"], default="MLP", help="Welches Modell verwendet werden soll")

    parser.add_argument("--rf-n-estimators", type=int, default=100, help="Anzahl der Bäume für RandomForest")
    parser.add_argument("--rf-max-depth", default=None, help="Maximale Tiefe der Bäume (None für unlimitiert)")

    parser.add_argument("--mlp-hidden-layers", default="100,50", help="Hidden layers für MLP als Komma-getrennte Liste, z.B. '100,50'")
    parser.add_argument("--mlp-max-iter", type=int, default=1000, help="Maximale Iterationen für MLP")
    parser.add_argument("--mlp-activation", choices=["identity", "logistic", "tanh", "relu"], default="relu", help="Aktivierungsfunktion für MLP")
    parser.add_argument("--mlp-solver", choices=["lbfgs", "sgd", "adam"], default="adam", help="Solver für MLP")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    trainer = Trainer(args)
    trainer.run()