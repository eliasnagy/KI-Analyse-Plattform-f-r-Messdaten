from pathlib import Path

from c1_analyzer import C1Analyzer


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    data_dir = script_dir / "c1"

    analyzer = C1Analyzer(data_dir=data_dir, chunk_size=200_000)
    result = analyzer.run()

    if result["files_total"] == 0:
        print(f"Keine CSV-Dateien gefunden in: {data_dir}")
        return

    print(f"Dateien geladen: {result['files_total']}")
    print(f"Gesamtzeilen: {result['rows_total']}")
    print(f"Gesamtspalten (vereint): {len(result['all_columns'])}")
    print("\nSpaltennamen:")
    print(result["all_columns"])

    print("\nDatei-Uebersicht (erste 10):")
    print(result["files_info"].head(10).to_string(index=False))

    numeric_summary = result["numeric_summary"]
    if numeric_summary.empty:
        print("\nKeine numerischen Spalten fuer eine Gesamtauswertung gefunden.")
    else:
        print("\nNumerische Gesamtauswertung:")
        print(numeric_summary.to_string())


if __name__ == "__main__":
    main()
