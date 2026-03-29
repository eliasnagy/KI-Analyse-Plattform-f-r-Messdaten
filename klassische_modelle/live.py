import argparse
import csv
import os
import time
from datetime import datetime
from glob import glob
from typing import List, Optional, Sequence, Tuple

import joblib
import numpy as np

from config import Config


def resolve_latest_model_and_scaler(model_type: str, model_path: Optional[str], scaler_path: Optional[str]) -> Tuple[str, Optional[str]]:
    output_dir = Config.OUTPUT_FILES_FOLDER

    if model_path:
        resolved_model = model_path
    else:
        pattern = os.path.join(output_dir, f"model_{model_type}_*.pkl")
        candidates = sorted(glob(pattern), key=os.path.getmtime, reverse=True)
        if not candidates:
            raise FileNotFoundError(f"Kein Modell gefunden fuer {model_type}: {pattern}")
        resolved_model = candidates[0]

    if scaler_path:
        resolved_scaler = scaler_path
    else:
        pattern = os.path.join(output_dir, f"scaler_{model_type}_*.pkl")
        candidates = sorted(glob(pattern), key=os.path.getmtime, reverse=True)
        resolved_scaler = candidates[0] if candidates else None

    return resolved_model, resolved_scaler


def _normalize_column_name(name: str) -> str:
    return name.strip().lstrip("\ufeff")


def _is_data_row(row: Sequence[str]) -> bool:
    """Prüfe ob eine Zeile numerische Daten ist (keine Text-Header)."""
    if not row or all(part.strip() == "" for part in row):
        return False
    for part in row:
        try:
            float(part.strip().lstrip("\ufeff"))
        except ValueError:
            return False
    return True


def validate_and_map_columns(header: Sequence[str], required_columns: Sequence[str]) -> List[str]:
    header_set = {_normalize_column_name(h) for h in header}
    missing = [col for col in required_columns if col not in header_set]
    if missing:
        raise ValueError(f"CSV-Header enthaelt nicht alle benoetigten Spalten. Fehlt: {missing}")

    # Reihenfolge muss dem Training entsprechen.
    return list(required_columns)


def build_features_from_single_row(sensor_values: Sequence[float], expected_features: int) -> np.ndarray:
    base = np.asarray(sensor_values, dtype=np.float64)
    base_feature_count = int(base.shape[0])

    if expected_features == base_feature_count:
        return base.reshape(1, -1)

    # Kompatibilitaet fuer Modelle, die auf 5 Statistikwerten pro Kanal trainiert wurden,
    # aber trotzdem nur die aktuelle Einzelzeile nutzen sollen (ohne Akkumulation).
    if expected_features == base_feature_count * 5:
        features: List[float] = []
        for value in base:
            features.extend([float(value), 0.0, float(value), float(value), float(abs(value))])
        return np.asarray(features, dtype=np.float64).reshape(1, -1)

    raise ValueError(
        f"Modell erwartet {expected_features} Features, CSV liefert {base_feature_count}. "
        "Unterstuetzt sind derzeit direkte 1:1-Features oder 5x-Statistikfeatures pro Kanal."
    )


def parse_sensor_row(raw_row: Sequence[str], column_order: Sequence[str], mapped_columns: Sequence[str]) -> List[float]:
    """Parse sensor values from CSV row.
    
    Wenn column_order == mapped_columns (headerless case), direkt zu float konvertieren.
    Sonst ein Dict-basiertes Mapping nutzen (mit Headers).
    """
    if column_order == mapped_columns:
        # Headerless: direkt parsen
        try:
            return [float(val.strip().lstrip("\ufeff")) for val in raw_row]
        except ValueError as exc:
            raise ValueError(f"Nicht-numerischer Wert in Zeile: {raw_row}") from exc
    else:
        # Mit Header: Dict-basiertes Mapping
        row_dict = {_normalize_column_name(col): value for col, value in zip(column_order, raw_row)}
        try:
            return [float(row_dict[col]) for col in mapped_columns]
        except (ValueError, KeyError) as exc:
            raise ValueError(f"Nicht-numerischer Wert in Zeile: {row_dict}") from exc


def predict_live(
    model,
    scaler,
    input_dir: str,
    log_csv: str,
    sleep_seconds: float,
    wear_threshold: Optional[float],
) -> None:
    os.makedirs(os.path.dirname(log_csv) or ".", exist_ok=True)

    required_columns = [c.strip() for c in Config.COLUMN_NAMES]
    expected_features = int(getattr(model, "n_features_in_", len(required_columns)))

    def process_file(file_path: str, start_index: int, tail_mode: bool) -> Tuple[int, bool]:
        with open(file_path, "r", encoding="utf-8", newline="") as file_handle:
            first_line = file_handle.readline()
            if not first_line:
                print(f"WARNUNG: Leere CSV-Datei uebersprungen: {file_path}")
                return start_index, False

            dialect = csv.Sniffer().sniff(first_line)
            first_row = next(csv.reader([first_line], dialect=dialect))

            if _is_data_row(first_row):
                print(f"Keine Header-Zeile gefunden. Nutze Config-Spalten: {', '.join(required_columns)}")
                mapped_columns = required_columns
                column_order = mapped_columns
                file_handle.seek(0)
            else:
                column_order = first_row
                mapped_columns = validate_and_map_columns(column_order, required_columns)

            if not os.path.exists(log_csv):
                with open(log_csv, "w", encoding="utf-8", newline="") as log_file:
                    writer = csv.writer(log_file)
                    header_row = ["timestamp", "sample_index"] + mapped_columns + ["prediction", "source_file", "alarm"]
                    writer.writerow(header_row)

            print(f"\nAuswertung startet fuer: {file_path}")
            sample_index = start_index

            while True:
                current_position = file_handle.tell()
                line = file_handle.readline()

                if not line:
                    if tail_mode:
                        time.sleep(sleep_seconds)
                        file_handle.seek(current_position)
                        continue
                    break

                # Bei unvollstaendigen Writes warten wir auf die vollstaendige Zeile.
                if tail_mode and (not line.endswith("\n") and not line.endswith("\r")):
                    time.sleep(sleep_seconds)
                    file_handle.seek(current_position)
                    continue

                parsed_line = next(csv.reader([line], dialect=dialect), None)
                if not parsed_line or all(part.strip() == "" for part in parsed_line):
                    continue

                if len(parsed_line) != len(mapped_columns):
                    print(f"WARNUNG: Zeile mit unpassender Spaltenanzahl uebersprungen: {parsed_line}")
                    continue

                try:
                    sensor_values = parse_sensor_row(parsed_line, column_order, mapped_columns)
                except ValueError as exc:
                    print(f"WARNUNG: {exc}")
                    continue

                sample_index += 1
                features = build_features_from_single_row(sensor_values, expected_features)
                if scaler is not None:
                    features = scaler.transform(features)

                prediction = float(model.predict(features)[0])
                timestamp = datetime.now().isoformat(timespec="seconds")
                alarm = wear_threshold is not None and prediction >= wear_threshold

                sensor_str = " ".join([f"{name}={val:.4f}" for name, val in zip(mapped_columns, sensor_values)])
                print(f"[{timestamp}] sample={sample_index:06d} {sensor_str} prediction={prediction:.6f}")

                with open(log_csv, "a", encoding="utf-8", newline="") as log_file:
                    writer = csv.writer(log_file)
                    row = [timestamp, sample_index] + sensor_values + [prediction, file_path, int(alarm)]
                    writer.writerow(row)

                if alarm:
                    print("=" * 72)
                    print(f"ALARM: Verschleiss-Grenzwert erreicht/ueberschritten! ({prediction:.6f} >= {wear_threshold:.6f})")
                    print("Bohrer wechseln. Live-Betrieb wird beendet.")
                    print("=" * 72)
                    return sample_index, True

                time.sleep(sleep_seconds)

            return sample_index, False

    print("Starte Live-Modus (Ordnerauswertung). Beenden mit CTRL+C.")
    print(f"Log:   {log_csv}")
    print(f"Sleep: {sleep_seconds}s")
    if wear_threshold is not None:
        print(f"Grenzwert: {wear_threshold}")
    else:
        print("Grenzwert: deaktiviert")

    sample_index = 0
    pattern = os.path.join(input_dir, "*.csv")
    csv_files = sorted(glob(pattern))
    if not csv_files:
        raise FileNotFoundError(f"Keine CSV-Dateien im Ordner gefunden: {input_dir}")

    print(f"Modus: Ordnerauswertung ({len(csv_files)} Dateien)")
    for file_path in csv_files:
        sample_index, should_stop = process_file(file_path, sample_index, tail_mode=False)
        if should_stop:
            return
    print("Ordnerauswertung abgeschlossen.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Live-Modus fuer klassische Modelle (RandomForest/MLP)")
    parser.add_argument("--model", choices=["RandomForest", "MLP"], required=True, help="Modell-Typ")
    parser.add_argument(
        "--input-dir",
        default=os.getenv("LIVE_INPUT_DIR", "").strip() or None,
        help="Ordner mit CSV-Dateien fuer Auswertung (z. B. ./live_daten/c2)",
    )
    parser.add_argument(
        "--log-csv",
        default=os.getenv("LIVE_LOG_CSV", "./output_files/live_predictions.csv"),
        help="CSV-Datei fuer Vorhersage-Log",
    )
    parser.add_argument(
        "--sleep-seconds",
        type=float,
        default=float(os.getenv("LIVE_SLEEP_SECONDS", "0.5")),
        help="Pause zwischen Zeilen in Sekunden",
    )
    threshold_env = os.getenv("LIVE_WEAR_THRESHOLD", "").strip()
    threshold_default = float(threshold_env) if threshold_env else None
    parser.add_argument(
        "--wear-threshold",
        type=float,
        default=threshold_default,
        help="Grenzwert fuer Verschleiss. Bei prediction >= threshold wird Alarm ausgeloest und beendet.",
    )
    parser.add_argument("--model-path", default=None, help="Expliziter Pfad zur Modell-PKL")
    parser.add_argument("--scaler-path", default=None, help="Expliziter Pfad zur Scaler-PKL")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.sleep_seconds < 0:
        raise ValueError("--sleep-seconds muss >= 0 sein")
    if args.wear_threshold is not None and args.wear_threshold < 0:
        raise ValueError("--wear-threshold muss >= 0 sein")

    if not args.input_dir:
        raise ValueError("--input-dir ist erforderlich. Einzeldatei-Modus wurde entfernt.")
    if not os.path.isdir(args.input_dir):
        raise FileNotFoundError(f"Input-Ordner nicht gefunden: {args.input_dir}")

    model_path, scaler_path = resolve_latest_model_and_scaler(args.model, args.model_path, args.scaler_path)

    print(f"Lade Modell: {model_path}")
    model = joblib.load(model_path)

    scaler = None
    if scaler_path:
        print(f"Lade Scaler: {scaler_path}")
        scaler = joblib.load(scaler_path)
    else:
        print("Kein Scaler gefunden/angegeben. Inferenz laeuft ohne Skalierung.")

    predict_live(
        model=model,
        scaler=scaler,
        input_dir=args.input_dir,
        log_csv=args.log_csv,
        sleep_seconds=args.sleep_seconds,
        wear_threshold=args.wear_threshold,
    )


if __name__ == "__main__":
    main()
