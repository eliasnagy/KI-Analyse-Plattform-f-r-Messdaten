from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import math
from typing import Any

import pandas as pd


@dataclass
class ColumnStats:
    count: int = 0
    sum_value: float = 0.0
    sum_squares: float = 0.0
    min_value: float = math.inf
    max_value: float = -math.inf

    def update(self, values: pd.Series) -> None:
        numeric_values = pd.to_numeric(values, errors="coerce").dropna()
        if numeric_values.empty:
            return

        current_count = int(numeric_values.count())
        current_sum = float(numeric_values.sum())
        current_sq = float((numeric_values * numeric_values).sum())
        current_min = float(numeric_values.min())
        current_max = float(numeric_values.max())

        self.count += current_count
        self.sum_value += current_sum
        self.sum_squares += current_sq
        self.min_value = min(self.min_value, current_min)
        self.max_value = max(self.max_value, current_max)

    def to_row(self) -> dict[str, float]:
        if self.count == 0:
            return {"count": 0.0, "mean": float("nan"), "std": float("nan"), "min": float("nan"), "max": float("nan")}

        mean = self.sum_value / self.count
        variance = (self.sum_squares / self.count) - (mean * mean)
        std = math.sqrt(max(variance, 0.0))

        return {
            "count": float(self.count),
            "mean": mean,
            "std": std,
            "min": self.min_value,
            "max": self.max_value,
        }


@dataclass
class C1Analyzer:
    data_dir: Path
    chunk_size: int = 200_000
    _column_stats: dict[str, ColumnStats] = field(default_factory=dict)
    _rows_total: int = 0
    _files_info: list[dict[str, Any]] = field(default_factory=list)
    _all_columns: set[str] = field(default_factory=set)

    def csv_files(self) -> list[Path]:
        return sorted(self.data_dir.glob("*.csv"))

    def run(self) -> dict[str, Any]:
        files = self.csv_files()
        if not files:
            return {
                "files_total": 0,
                "rows_total": 0,
                "all_columns": [],
                "files_info": pd.DataFrame(),
                "numeric_summary": pd.DataFrame(),
            }

        for csv_path in files:
            self._process_file(csv_path)

        return {
            "files_total": len(files),
            "rows_total": self._rows_total,
            "all_columns": sorted(self._all_columns),
            "files_info": pd.DataFrame(self._files_info),
            "numeric_summary": self._build_numeric_summary(),
        }

    def _process_file(self, csv_path: Path) -> None:
        rows_in_file = 0
        columns_in_file: set[str] = set()

        for chunk in pd.read_csv(csv_path, chunksize=self.chunk_size):
            rows_in_chunk = int(len(chunk))
            rows_in_file += rows_in_chunk
            self._rows_total += rows_in_chunk

            columns_in_chunk = set(chunk.columns.tolist())
            columns_in_file.update(columns_in_chunk)
            self._all_columns.update(columns_in_chunk)

            numeric_chunk = chunk.select_dtypes(include="number")
            for col_name, series in numeric_chunk.items():
                if col_name not in self._column_stats:
                    self._column_stats[col_name] = ColumnStats()
                self._column_stats[col_name].update(series)

        self._files_info.append(
            {
                "file": csv_path.name,
                "rows": rows_in_file,
                "columns": len(columns_in_file),
            }
        )

    def _build_numeric_summary(self) -> pd.DataFrame:
        rows: dict[str, dict[str, float]] = {}
        for col_name, stats in self._column_stats.items():
            rows[col_name] = stats.to_row()

        if not rows:
            return pd.DataFrame()

        summary = pd.DataFrame.from_dict(rows, orient="index")
        summary.index.name = "column"
        return summary.sort_index()
