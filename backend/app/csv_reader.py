"""
SyntheticRows — robust CSV reading.

Single entry point: read_csv_safely(raw_bytes, filename) -> pd.DataFrame
Raises CSVReadError(message) with a clean, human-readable message on any failure.
Handles: encoding fallback, delimiter sniffing, empty / headers-only files,
malformed structure, wrong-delimiter single-column blobs, duplicate columns,
and files that are secretly Excel/HTML/JSON renamed to .csv.
"""
import io
import csv as _csv
import pandas as pd


class CSVReadError(Exception):
    """Raised when a CSV cannot be read into a usable DataFrame."""
    pass


# Encodings we try in order. utf-8-sig first to transparently strip a BOM.
_ENCODINGS = ["utf-8-sig", "utf-8", "cp1252", "latin-1"]

# Delimiters we consider when sniffing.
_DELIMITERS = [",", ";", "\t", "|"]


def _decode(raw: bytes) -> str:
    """Decode bytes to text, trying common encodings. latin-1 never fails,
    so it's the guaranteed last resort."""
    if raw is None or len(raw) == 0:
        raise CSVReadError("This file is empty. Please upload a CSV that contains data.")

    # Detect obviously-not-CSV binary signatures (real Excel .xlsx is a zip; .xls is OLE).
    head = raw[:8]
    if head[:2] == b"PK":  # zip / xlsx / docx
        raise CSVReadError(
            "This looks like an Excel (.xlsx) file, not a CSV. "
            "Please open it in Excel or Sheets and use 'Save As → CSV', then upload that."
        )
    if head[:4] == b"\xd0\xcf\x11\xe0":  # OLE header / old .xls
        raise CSVReadError(
            "This looks like an old Excel (.xls) file, not a CSV. "
            "Please re-save it as CSV and upload that."
        )

    for enc in _ENCODINGS:
        try:
            return raw.decode(enc)
        except (UnicodeDecodeError, LookupError):
            continue
    # Should be unreachable because latin-1 maps every byte, but just in case:
    raise CSVReadError("We couldn't read this file's text encoding. Try re-saving it as a UTF-8 CSV.")


def _looks_like_html_or_json(text: str) -> bool:
    t = text.lstrip()[:200].lower()
    return t.startswith("<!doctype") or t.startswith("<html") or t.startswith("<") and "table" in t


def _sniff_delimiter(text: str) -> str:
    """Pick the most likely delimiter from the first few non-empty lines.
    Falls back to comma."""
    sample = "\n".join([ln for ln in text.splitlines() if ln.strip()][:20])
    if not sample:
        return ","
    # Try Python's sniffer first.
    try:
        dialect = _csv.Sniffer().sniff(sample, delimiters="".join(_DELIMITERS))
        if dialect.delimiter in _DELIMITERS:
            return dialect.delimiter
    except Exception:
        pass
    # Fallback: count candidates on the header line, pick the most frequent.
    header = sample.splitlines()[0]
    counts = {d: header.count(d) for d in _DELIMITERS}
    best = max(counts, key=counts.get)
    return best if counts[best] > 0 else ","


def read_csv_safely(raw: bytes, filename: str = "file.csv") -> pd.DataFrame:
    text = _decode(raw)

    if not text.strip():
        raise CSVReadError("This file is empty. Please upload a CSV that contains data.")

    if _looks_like_html_or_json(text):
        raise CSVReadError(
            "This file doesn't look like a CSV (it appears to be HTML or another format). "
            "Please upload a plain CSV file."
        )

    delimiter = _sniff_delimiter(text)

    # Parse. python engine tolerates ragged rows better and supports on_bad_lines.
    try:
        df = pd.read_csv(
            io.StringIO(text),
            sep=delimiter,
            engine="python",
            skip_blank_lines=True,
            on_bad_lines="skip",  # drop malformed rows instead of crashing
        )
    except pd.errors.EmptyDataError:
        raise CSVReadError("This file has no readable rows or columns. Please check the file and try again.")
    except Exception:
        # Last-ditch: let pandas auto-sniff the separator itself.
        try:
            df = pd.read_csv(io.StringIO(text), sep=None, engine="python", on_bad_lines="skip")
        except Exception:
            raise CSVReadError(
                "We couldn't parse this file as a CSV. It may be malformed or use an unusual format. "
                "Try re-saving it as a standard comma-separated CSV."
            )

    # ---- Validate the result is actually usable ----

    if df.shape[1] == 0:
        raise CSVReadError("No columns were found in this file. Please upload a valid CSV.")

    # A single column when the text clearly contains other delimiters usually means
    # the delimiter was wrong (e.g. a semicolon file read as one column).
    if df.shape[1] == 1:
        first_cell = str(df.columns[0])
        if any(d in first_cell for d in _DELIMITERS):
            raise CSVReadError(
                "We couldn't detect the column separator in this file. "
                "Please make sure it's comma-separated and try again."
            )

    if df.shape[0] == 0:
        raise CSVReadError(
            "This file has column headers but no data rows. "
            "Please upload a CSV that contains at least one row of data."
        )

    # Clean up duplicate / unnamed columns so downstream code is safe.
    df = _normalize_columns(df)

    return df


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Make column names safe and unique: strip whitespace, name blanks,
    and de-duplicate repeated names so lookups never collide."""
    new_cols = []
    seen = {}
    for i, col in enumerate(df.columns):
        name = str(col).strip()
        if name == "" or name.lower().startswith("unnamed"):
            name = f"column_{i + 1}"
        if name in seen:
            seen[name] += 1
            name = f"{name}_{seen[name]}"
        else:
            seen[name] = 0
        new_cols.append(name)
    df.columns = new_cols
    return df