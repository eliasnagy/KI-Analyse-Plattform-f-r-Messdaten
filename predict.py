import argparse
import os
import sys
import joblib
import onnxruntime as ort
import numpy as np

from config import Config
from data_processing import extract_features


def find_latest_model_file(folder, prefix):
    # Prefer ONNX if available, otherwise fall back to .pkl
    onnx_files = [f for f in os.listdir(folder) if f.startswith(prefix) and f.endswith('.onnx')]
    if onnx_files:
        onnx_files = sorted(onnx_files)
        return os.path.join(folder, onnx_files[-1])

    pkl_files = [f for f in os.listdir(folder) if f.startswith(prefix) and f.endswith('.pkl')]
    if pkl_files:
        pkl_files = sorted(pkl_files)
        return os.path.join(folder, pkl_files[-1])

    return None


def find_latest_file(folder, prefix, suffix='.pkl'):
    files = [f for f in os.listdir(folder) if f.startswith(prefix) and f.endswith(suffix)]
    if not files:
        return None
    files = sorted(files)
    return os.path.join(folder, files[-1])


def load_model_and_scaler(model_path, scaler_path=None):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Modell nicht gefunden: {model_path}")

    # ONNX model
    if model_path.endswith('.onnx'):
        sess = ort.InferenceSession(model_path)
        return sess, None

    # sklearn model (.pkl)
    if not scaler_path or not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Scaler nicht gefunden: {scaler_path}")

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler


def predict_single_file(model, scaler, csv_path):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Input-Datei nicht gefunden: {csv_path}")

    features = extract_features(csv_path)
    X = np.array(features).reshape(1, -1)
    # ONNX runtime
    if isinstance(model, ort.InferenceSession):
        inp_name = model.get_inputs()[0].name
        pred = model.run(None, {inp_name: X.astype(np.float32)})[0]
        return float(pred[0])

    # sklearn
    X_scaled = scaler.transform(X)
    pred = model.predict(X_scaled)
    return float(pred[0])


def predict_folder(model, scaler, folder_path):
    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"Input-Ordner nicht gefunden: {folder_path}")
    files = [f for f in os.listdir(folder_path) if f.lower().endswith('.csv')]
    files = sorted(files)
    results = []
    for f in files:
        path = os.path.join(folder_path, f)
        try:
            pred = predict_single_file(model, scaler, path)
            results.append((f, pred))
            print(f"✓ {f}: {pred:.6f}")
        except Exception as e:
            print(f"Fehler für {f}: {e}")
    return results


def parse_args():
    parser = argparse.ArgumentParser(description="Vorhersage (Produktivbetrieb) für eine einzelne Messdatei (CSV)")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--input-file", help="Pfad zur neuen CSV-Datei (ein 'Schnitt')")
    group.add_argument("--input-folder", help="Pfad zu einem Ordner mit CSV-Dateien für Stapelvorhersage")
    parser.add_argument("--model-path", default=None, help="Pfad zur Modell-.pkl Datei (optional, Standard: neuestes im output folder)")
    parser.add_argument("--scaler-path", default=None, help="Pfad zur Scaler-.pkl Datei (optional, Standard: neuestes im output folder)")
    parser.add_argument("--output", default=None, help="Optional: Pfad, um Vorhersage als Textdatei zu speichern")
    return parser.parse_args()


def main():
    args = parse_args()
    Config.print_config()

    output_folder = Config.OUTPUT_FILES_FOLDER
    os.makedirs(output_folder, exist_ok=True)

    model_path = args.model_path
    scaler_path = args.scaler_path

    if model_path is None:
        model_path = find_latest_model_file(output_folder, "model_")
        if model_path is None:
            print(f"Kein Modell in {output_folder} gefunden. Bitte --model-path angeben.")
            sys.exit(1)
        print(f"Automatisch neuestes Modell gefunden: {model_path}")

    if scaler_path is None:
        scaler_path = find_latest_file(output_folder, "scaler_", ".pkl")
        # If model is ONNX (pipeline includes scaler), scaler is optional
        if scaler_path is None:
            print("Kein Scaler automatisch gefunden. Falls Modell ONNX ist, ist das OK.")
        else:
            print(f"Automatisch neuesten Scaler gefunden: {scaler_path}")

    print(f"Lade Modell: {model_path}")
    if scaler_path:
        print(f"Lade Scaler: {scaler_path}")
    model, scaler = load_model_and_scaler(model_path, scaler_path)

    print(f"Extrahiere Features aus: {args.input_file} ...")
    # Single file or folder handling
    if args.input_file:
        print(f"Extrahiere Features aus: {args.input_file} ...")
        try:
            pred = predict_single_file(model, scaler, args.input_file)
        except Exception as e:
            print(f"Fehler bei Vorhersage: {e}")
            sys.exit(2)
        print(f"✓ Vorhersage (max_wear) für '{args.input_file}': {pred:.4f}")
        if args.output:
            with open(args.output, 'w', encoding='utf-8') as fh:
                fh.write(f"{pred:.6f}\n")
            print(f"Vorhersage in {args.output} gespeichert")
    else:
        # folder mode
        print(f"Starte Stapelvorhersage für Ordner: {args.input_folder}")
        results = predict_folder(model, scaler, args.input_folder)
        if args.output:
            # write combined CSV: filename,prediction
            try:
                with open(args.output, 'w', encoding='utf-8') as fh:
                    fh.write('filename,prediction\n')
                    for fname, val in results:
                        fh.write(f"{fname},{val:.6f}\n")
                print(f"Alle Vorhersagen in {args.output} gespeichert")
            except Exception as e:
                print(f"Fehler beim Speichern der Ausgabe: {e}")


if __name__ == "__main__":
    main()
