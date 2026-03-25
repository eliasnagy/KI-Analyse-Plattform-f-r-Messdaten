import os
import numpy as np
import joblib
import matplotlib.pyplot as plt
import csv

from config import Config
from data_processing import DatasetBuilder
from sklearn.ensemble import RandomForestRegressor


def find_latest_file(folder, prefix):
    files = [f for f in os.listdir(folder) if f.startswith(prefix) and f.endswith('.pkl')]
    if not files:
        return None
    files = sorted(files)
    return os.path.join(folder, files[-1])


def outlier_report_and_winsorize(y):
    q1, q3 = np.percentile(y, [25, 75])
    iqr = q3 - q1
    low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    mask = (y < low) | (y > high)
    report = {
        'count': int(mask.sum()),
        'low_bound': float(low),
        'high_bound': float(high),
        'min': float(np.min(y)),
        'max': float(np.max(y)),
        'mean': float(np.mean(y)),
        'median': float(np.median(y)),
    }
    y_winsor = np.clip(y, low, high)
    return report, y_winsor, mask


def save_report_csv(path, rows, headers):
    with open(path, 'w', newline='', encoding='utf-8') as fh:
        writer = csv.DictWriter(fh, fieldnames=headers)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def main():
    Config._setup_paths()
    input_folders = [os.path.join(Config.BASE_TRAINING_DIR, f) for f in Config.INPUT_FOLDERS]

    print("Lade/baue Dataset...")
    X, y = DatasetBuilder.build_or_load_dataset(input_folders, save_name='combined_features')

    print("Erstelle Outlier-Report und winsorize y...")
    report, y_winsor, mask = outlier_report_and_winsorize(y)
    os.makedirs(Config.OUTPUT_FILES_FOLDER, exist_ok=True)
    os.makedirs(Config.NUMPY_FILES_FOLDER, exist_ok=True)

    # Save winsorized y
    winsor_path = f"{Config.NUMPY_FILES_FOLDER}/combined_features_y_winsorized.npy"
    np.save(winsor_path, y_winsor)
    print(f"✓ Winsorized y gespeichert: {winsor_path}")

    # Write a small summary CSV of outliers
    outlier_rows = []
    idxs = np.nonzero(mask)[0]
    for i in idxs:
        outlier_rows.append({'index': int(i), 'y': float(y[i])})
    save_report_csv(f"{Config.OUTPUT_FILES_FOLDER}/outliers_report.csv", outlier_rows, ['index', 'y'])
    print(f"✓ Outlier-Report gespeichert: {Config.OUTPUT_FILES_FOLDER}/outliers_report.csv (count={len(outlier_rows)})")

    # Feature importances via RandomForest (fit on original X)
    print("Fitte RandomForest für Feature-Importances...")
    rf = RandomForestRegressor(n_estimators=200, random_state=Config.RANDOM_STATE)
    try:
        rf.fit(X, y_winsor)
    except Exception:
        rf.fit(X, y)  # fallback

    importances = rf.feature_importances_
    fig_path = f"{Config.OUTPUT_FILES_FOLDER}/feature_importances.png"
    plt.figure(figsize=(10, 4))
    plt.bar(range(len(importances)), importances)
    plt.xlabel('feature_index')
    plt.ylabel('importance')
    plt.title('Feature Importances (RandomForest)')
    plt.tight_layout()
    plt.savefig(fig_path)
    plt.close()
    print(f"✓ Feature importances plot: {fig_path}")

    # Residuals: try to use latest saved model, else use RF
    model_path = find_latest_file(Config.OUTPUT_FILES_FOLDER, 'model_')
    scaler_path = find_latest_file(Config.OUTPUT_FILES_FOLDER, 'scaler_')
    if model_path:
        print(f"Lade gespeichertes Modell: {model_path}")
        model = joblib.load(model_path)
        if scaler_path:
            scaler = joblib.load(scaler_path)
            X_for_pred = scaler.transform(X)
        else:
            X_for_pred = X
        preds = model.predict(X_for_pred)
    else:
        print("Kein gespeichertes Modell gefunden — verwende RandomForest predictions")
        preds = rf.predict(X)

    residuals = preds - y
    res_fig = f"{Config.OUTPUT_FILES_FOLDER}/residuals_vs_pred.png"
    plt.figure(figsize=(6, 4))
    plt.scatter(preds, residuals, s=6)
    plt.axhline(0, color='r', linestyle='--')
    plt.xlabel('predicted')
    plt.ylabel('residual (pred - true)')
    plt.title('Residuals vs Predictions')
    plt.tight_layout()
    plt.savefig(res_fig)
    plt.close()
    print(f"✓ Residuals plot: {res_fig}")

    # Histogram of residuals
    hist_fig = f"{Config.OUTPUT_FILES_FOLDER}/residuals_hist.png"
    plt.figure(figsize=(6, 4))
    plt.hist(residuals, bins=80)
    plt.xlabel('residual')
    plt.ylabel('count')
    plt.title('Residuals Distribution')
    plt.tight_layout()
    plt.savefig(hist_fig)
    plt.close()
    print(f"✓ Residuals histogram: {hist_fig}")

    # Save feature importances CSV
    fi_rows = [{'feature_index': i, 'importance': float(v)} for i, v in enumerate(importances)]
    save_report_csv(f"{Config.OUTPUT_FILES_FOLDER}/feature_importances.csv", fi_rows, ['feature_index', 'importance'])
    print(f"✓ Feature importances CSV: {Config.OUTPUT_FILES_FOLDER}/feature_importances.csv")


if __name__ == '__main__':
    main()
