"""
Data ingestion and schema discovery.

Two very different retail schemas have to feed the same application:

* the public Pentaho *sales_data_sample* file (one row per order line, columns
  such as ``ORDERDATE, SALES, QUANTITYORDERED, PRICEEACH, PRODUCTLINE, STATUS``);
* a merged Walmart-style weekly file (``date, weekly_sales, store_id,
  department, is_holiday, temperature, ...``).

Rather than hard-coding either layout, this module inspects whatever it is given,
normalises the headers and resolves them against the alias tables in
``config.COLUMN_ALIASES``. Anything it cannot map is reported, never guessed.
"""

from __future__ import annotations

import csv
import io
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO, Dict, Iterable, List, Optional, Tuple
from urllib.error import URLError
from urllib.request import Request, urlopen

import pandas as pd

import config


class DataSourceError(RuntimeError):
    """Raised when a dataset cannot be read at all."""


class SchemaError(ValueError):
    """Raised when a dataset does not contain the columns the app requires."""


# --------------------------------------------------------------------------- #
# Header handling
# --------------------------------------------------------------------------- #
def normalise_header(name: object) -> str:
    """Lowercase a header and keep alphanumerics only.

    ``"Weekly_Sales"``, ``"weekly sales"`` and ``"WEEKLY-SALES"`` all collapse to
    ``"weeklysales"`` so alias matching is insensitive to punctuation and case.
    """
    return "".join(ch for ch in str(name).strip().lower() if ch.isalnum())


def _alias_index(canonical: str, header: str) -> int:
    """Position of *header* in the alias list; lower means higher priority."""
    aliases = [normalise_header(a) for a in config.COLUMN_ALIASES.get(canonical, ())]
    try:
        return aliases.index(header)
    except ValueError:
        return len(aliases)


@dataclass
class SchemaMap:
    """Result of resolving a CSV's headers against the canonical model."""

    mapping: Dict[str, str] = field(default_factory=dict)   # canonical -> header
    unmapped: List[str] = field(default_factory=list)       # ignored headers
    conflicts: List[str] = field(default_factory=list)      # duplicate mappings
    missing_required: List[str] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        return not self.missing_required

    def describe(self) -> Dict[str, str]:
        """Canonical field -> original header, for display."""
        return dict(self.mapping)


def detect_schema(columns: Iterable[object]) -> SchemaMap:
    """Resolve *columns* against :data:`config.COLUMN_ALIASES`.

    A canonical field is mapped at most once. When several headers could serve
    the same field (a file containing both ``date`` and ``orderdate``), the
    alias order in the config decides, and the rejected candidate is recorded
    as a conflict instead of being silently dropped.
    """
    original = [str(c) for c in columns]
    lookup: Dict[str, List[Tuple[str, int]]] = {}
    for col in original:
        lookup.setdefault(normalise_header(col), []).append((col, 0))

    result = SchemaMap()
    claimed: set[str] = set()
    for canonical, aliases in config.COLUMN_ALIASES.items():
        # Keyed by column name so that several aliases normalising to the same
        # string ("order_date" and "ORDERDATE") cannot make a column compete
        # with itself.
        candidates: Dict[str, int] = {}
        for alias in aliases:
            norm = normalise_header(alias)
            for col, _ in lookup.get(norm, []):
                if col in claimed:
                    continue
                index = _alias_index(canonical, norm)
                candidates[col] = min(candidates.get(col, len(aliases)), index)
        if not candidates:
            continue
        winner = min(candidates, key=lambda name: candidates[name])
        result.mapping[canonical] = winner
        claimed.add(winner)
        for loser in candidates:
            if loser != winner:
                result.conflicts.append(
                    f"'{loser}' also looked like '{canonical}' but '{winner}' was used"
                )

    result.unmapped = [c for c in original if c not in claimed]
    result.missing_required = [
        f for f in config.REQUIRED_FIELDS if f not in result.mapping
    ]
    return result


# --------------------------------------------------------------------------- #
# Reading
# --------------------------------------------------------------------------- #
@dataclass
class LoadedData:
    """A dataframe plus everything we know about where it came from."""

    frame: pd.DataFrame
    source_label: str
    encoding: str
    delimiter: str
    row_count: int
    column_count: int

    def summary(self) -> Dict[str, object]:
        return {
            "Source": self.source_label,
            "Encoding": self.encoding.upper(),
            "Delimiter": repr(self.delimiter),
            "Rows": f"{self.row_count:,}",
            "Columns": self.column_count,
        }


def _sniff_delimiter(sample: str) -> str:
    try:
        return csv.Sniffer().sniff(sample, delimiters=",;\t|").delimiter
    except Exception:  # noqa: BLE001 - sniffer is best-effort only
        return ","


def _read_text(raw: bytes) -> Tuple[str, str]:
    """Decode bytes by trying every configured encoding."""
    last_error: Optional[Exception] = None
    for encoding in config.ENCODINGS:
        try:
            return raw.decode(encoding), encoding
        except UnicodeDecodeError as exc:  # try the next candidate
            last_error = exc
    raise DataSourceError(
        f"Could not decode the file with any of {config.ENCODINGS}. "
        f"Last error: {last_error}"
    )


def _parse_csv_text(text: str, source_label: str, encoding: str) -> LoadedData:
    delimiter = _sniff_delimiter(text[:65536])
    try:
        frame = pd.read_csv(
            io.StringIO(text),
            sep=delimiter,
            dtype=str,            # keep raw text: typing happens in preprocessing
            keep_default_na=True,
            na_values=["", "NA", "N/A", "na", "null", "NULL", "None", "?"],
            encoding_errors="replace",
        )
    except Exception as exc:  # noqa: BLE001
        raise DataSourceError(f"Pandas could not parse {source_label}: {exc}") from exc

    if frame.empty:
        raise DataSourceError(f"{source_label} contains a header but no data rows.")
    if len(frame) > config.MAX_ROWS:
        raise DataSourceError(
            f"{source_label} has {len(frame):,} rows which exceeds the "
            f"{config.MAX_ROWS:,} row safety limit."
        )
    frame.columns = [str(c).strip() for c in frame.columns]
    return LoadedData(
        frame=frame,
        source_label=source_label,
        encoding=encoding,
        delimiter=delimiter,
        row_count=len(frame),
        column_count=frame.shape[1],
    )


def load_from_path(path: str | Path) -> LoadedData:
    """Read a CSV from disk, tolerating UTF-8, CP1252 and Latin-1."""
    p = Path(path)
    if not p.exists():
        raise DataSourceError(f"File not found: {p}")
    raw = p.read_bytes()
    text, encoding = _read_text(raw)
    return _parse_csv_text(text, p.name, encoding)


def load_from_bytes(raw: bytes, name: str = "uploaded.csv") -> LoadedData:
    """Read a CSV that arrived through the dashboard uploader."""
    if not raw:
        raise DataSourceError("The uploaded file is empty.")
    if len(raw) > config.MAX_UPLOAD_BYTES:
        raise DataSourceError(
            f"Upload is {len(raw) / 1e6:.1f} MB; the limit is "
            f"{config.MAX_UPLOAD_BYTES / 1e6:.0f} MB."
        )
    text, encoding = _read_text(raw)
    return _parse_csv_text(text, name, encoding)


def load_from_url(url: str, timeout: int = 45) -> LoadedData:
    """Download a CSV over HTTP(S)."""
    try:
        request = Request(url, headers={"User-Agent": "BizPredictAI/1.0"})
        with urlopen(request, timeout=timeout) as response:  # noqa: S310
            raw = response.read()
    except (URLError, TimeoutError, OSError) as exc:
        raise DataSourceError(
            f"Could not download the dataset ({exc}). Check your connection or "
            "load the bundled local copy instead."
        ) from exc
    text, encoding = _read_text(raw)
    label = url.rstrip("/").rsplit("/", 1)[-1]
    return _parse_csv_text(text, label, encoding)


def load_default_dataset(prefer_cache: bool = True) -> LoadedData:
    """Load the brief's default dataset, falling back to the bundled copy.

    The remote file is authoritative; the local copy exists so the demo still
    works offline. A warning is raised by the caller if the fallback was used.
    """
    if prefer_cache and config.DEFAULT_DATASET_LOCAL.exists():
        return load_from_path(config.DEFAULT_DATASET_LOCAL)
    try:
        return load_from_url(config.DEFAULT_DATASET_URL)
    except DataSourceError:
        if config.DEFAULT_DATASET_LOCAL.exists():
            return load_from_path(config.DEFAULT_DATASET_LOCAL)
        raise


def load_merged_dataset() -> LoadedData:
    """Load the optional 156k-row merged weekly dataset if it is bundled."""
    if not config.MERGED_DATASET_LOCAL.exists():
        raise DataSourceError(
            "The merged retail dataset is not present in data/. Upload it from "
            "the sidebar or add it as data/retail_sales_complete_dataset.csv."
        )
    return load_from_path(config.MERGED_DATASET_LOCAL)


# --------------------------------------------------------------------------- #
# Dataset profile
# --------------------------------------------------------------------------- #
@dataclass
class DatasetProfile:
    """Human-readable description of what was found in a dataset."""

    source_label: str
    schema: SchemaMap
    dtypes: Dict[str, str]
    row_count: int
    column_count: int
    frequency: str                     # "daily" or "weekly" (raw observation grid)
    has_order_ids: bool
    has_units: bool
    has_price: bool
    has_category: bool
    has_store: bool
    has_holiday_flag: bool
    has_holiday_names: bool
    context_columns: List[str]
    label_columns: List[str]
    warnings: List[str] = field(default_factory=list)

    def field_label(self, canonical: str) -> str:
        """Original header used for *canonical*, or a dash when unmapped."""
        return self.schema.mapping.get(canonical, "—")

    @property
    def is_weekly(self) -> bool:
        return self.frequency == "weekly"


def profile_dataset(loaded: LoadedData) -> DatasetProfile:
    """Inspect headers and raw values to describe the dataset before cleaning.

    Frequency is inferred from the *median spacing between distinct dates*,
    which distinguishes per-transaction data from an already weekly-aggregated
    file without relying on column names.
    """
    frame = loaded.frame
    schema = detect_schema(frame.columns)

    if not schema.is_valid:
        missing = ", ".join(schema.missing_required)
        raise SchemaError(
            f"'{loaded.source_label}' is missing required column(s): {missing}. "
            f"Available columns are: {', '.join(map(str, frame.columns))}. "
            "Rename the columns to something recognisable (for example "
            "'date' and 'sales' / 'weekly_sales') and try again."
        )

    date_col = schema.mapping["date"]
    dates = pd.to_datetime(frame[date_col], errors="coerce", format="mixed")
    frequency = _infer_frequency(dates)

    original_headers = [str(c) for c in frame.columns]
    context_columns = [
        c for c in original_headers
        if normalise_header(c) in {normalise_header(x) for x in config.EXTRA_CONTEXT_COLUMNS}
    ]
    label_columns = [
        c for c in original_headers
        if normalise_header(c) in {normalise_header(x) for x in config.EXTRA_LABEL_COLUMNS}
    ]

    warnings: List[str] = []
    if schema.conflicts:
        warnings.extend(schema.conflicts)
    for canonical in ("units", "unit_price", "category", "store", "order_id"):
        if canonical not in schema.mapping:
            warnings.append(
                f"No '{canonical}' column was recognised; features that need it "
                "are disabled rather than estimated."
            )
    if frequency == "weekly":
        warnings.append(
            "Dates are spaced ~7 days apart, so this is weekly data. Daily "
            "revenue charts and day-of-week seasonality are therefore not "
            "available for it."
        )

    return DatasetProfile(
        source_label=loaded.source_label,
        schema=schema,
        dtypes={str(c): _pandas_dtype_name(frame[c]) for c in frame.columns},
        row_count=loaded.row_count,
        column_count=loaded.column_count,
        frequency=frequency,
        has_order_ids="order_id" in schema.mapping,
        has_units="units" in schema.mapping,
        has_price="unit_price" in schema.mapping,
        has_category="category" in schema.mapping,
        has_store="store" in schema.mapping,
        has_holiday_flag="is_holiday" in schema.mapping,
        has_holiday_names="holiday_name" in schema.mapping,
        context_columns=context_columns,
        label_columns=label_columns,
        warnings=warnings,
    )


def _pandas_dtype_name(series: pd.Series) -> str:
    """Infer a friendly dtype for a raw (string) column."""
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().mean() >= 0.9:
        return "numeric"
    parsed = pd.to_datetime(series, errors="coerce", format="mixed")
    if parsed.notna().mean() >= 0.9:
        return "datetime"
    return "text"


def _infer_frequency(dates: pd.Series) -> str:
    """Return 'weekly' when distinct dates are spaced ~7 days apart."""
    unique = pd.Series(dates.dropna().unique()).sort_values()
    if len(unique) < 5:
        return "daily"
    gaps = unique.diff().dt.days.dropna()
    if gaps.empty:
        return "daily"
    median_gap = float(gaps.median())
    return "weekly" if 6.0 <= median_gap <= 8.0 else "daily"


# --------------------------------------------------------------------------- #
# Sample helper (used by the Data Explorer)
# --------------------------------------------------------------------------- #
def read_head(path: str | Path, n: int = 5) -> pd.DataFrame:
    """Read the first *n* rows of a CSV without loading the whole file."""
    with open(path, "r", encoding="utf-8", errors="replace") as handle:
        return pd.read_csv(handle, nrows=n)


def open_binary(path: str | Path) -> IO[bytes]:
    """Small convenience wrapper used by the download buttons."""
    return open(path, "rb")