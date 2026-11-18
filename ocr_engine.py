"""

ocr_engine.py -- Computer-vision module for the Smart Electricity Bill Analyzer.

Turns a photo of an MSEDCL (Maharashtra) LT-I domestic electricity bill into a

dictionary of structured fields, each paired with an OCR confidence score.

Pipeline (Steps 2-3 of the project's 7-step data pipeline):

    image bytes --> cv2.imdecode --> preprocess() --> run_ocr() --> parse_fields()

Design rules (from the project proposal):

    * No database and no UI dependencies -- only app.py imports this module.

    * OCR is a *suggestion engine*: low-confidence or missing fields come back

      with a low confidence (or 0.0) so app.py can flag them for human review.

Public API (frozen function contracts):

    preprocess(img_bgr)      -> np.ndarray

    run_ocr(img)             -> list[tuple[str, float]]

    parse_fields(ocr_lines)  -> dict[str, tuple[value, float]]

    extract_bill(image_bytes)-> dict[str, tuple[value, float]]

"""

from __future__ import annotations

import io

import logging

import re

import threading

from datetime import datetime

from typing import Any, Callable, Optional

import cv2

import easyocr

import numpy as np

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #

# Configuration

# --------------------------------------------------------------------------- #

#: Output format of ``bill_month``. "%Y-%m" matches the SQLite schema

#: (``billing_month`` is 'YYYY-MM'); switch to "%b %Y" for strings like "Aug 2024".

BILL_MONTH_FORMAT: str = "%Y-%m"

#: Very large phone photos are shrunk (never enlarged) before pre-processing so

#: that CPU-only OCR stays within the \~15 s demo budget. Longest side, in px.

MAX_INPUT_SIDE: int = 2200

#: Confidence multiplier applied when a value comes from a fallback heuristic

#: (not anchored to its keyword) or is derived from other fields.

FALLBACK_PENALTY: float = 0.7

#: When reading arithmetic (curr - prev == units, +/-1) fails, the three

#: confidences are capped below this value so the UI flags them for review.

INCONSISTENT_CONF_CAP: float = 0.49

_MONTHS: dict[str, int] = {

    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,

    "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,

}

# --------------------------------------------------------------------------- #

# Regular expressions

# --------------------------------------------------------------------------- #

# All patterns are keyword-anchored (not position-based), case-insensitive, and

# tolerant of common OCR slips ("Reacing", missing dots, extra spaces).

# Group 1 of every pattern is the *value*; its character span is used to look up

# the confidence of the OCR boxes that produced it. ``\D{0,N}`` deliberately

# crosses line breaks, because label and value are often separate OCR boxes.

_NUM = r"\d{1,3}(?:,\d{2,3})+(?:\.\d{1,2})?|\d+(?:\.\d{1,2})?"  # 1,234.50 / 12,34,567 / 98

# Characters OCR frequently confuses with digits (only used for consumer number).

_DIGITISH = r"[\dOoIl|]"

_PATTERNS: dict[str, list[re.Pattern[str]]] = {

    "consumer_no": [

        re.compile(

            r"(?:Consumer|Cons\.?)\s*(?:No|Number|Num|#)\b\.?\s*[:\-\.]?\s*"

            rf"({_DIGITISH}(?:[ \-]?{_DIGITISH}){{11}})(?!\d)",

            re.IGNORECASE,

        ),

    ],

    "units": [

        re.compile(

            r"(?:Total\s*)?Units?\s*Consum\w*\D{0,20}?(\d{1,5})(?![\d,.]\d)",

            re.IGNORECASE,

        ),

        re.compile(

            r"(?:Total|Net)\s*Consumption(?:\s*\(?\s*(?:kWh|units?)\s*\)?)?\D{0,20}?(\d{1,5})(?![\d,.]\d)",

            re.IGNORECASE,

        ),

        re.compile(

            r"Consumption\s*\(?\s*(?:kWh|units?)\s*\)?\D{0,15}?(\d{1,5})(?![\d,.]\d)",

            re.IGNORECASE,

        ),

    ],

    "prev_reading": [

        re.compile(

            r"(?:Previous|Prev\.?|Last|Old)\s*(?:Meter\s*)?Read\w{0,4}\D{0,15}?(\d{1,7})(?![\d,.]\d)",

            re.IGNORECASE,

        ),

    ],

    "curr_reading": [

        re.compile(

            r"(?:Current|Curr\.?|Present|New)\s*(?:Meter\s*)?Read\w{0,4}\D{0,15}?(\d{1,7})(?![\d,.]\d)",

            re.IGNORECASE,

        ),

    ],

    "total_amount": [

        re.compile(

            rf"\bBill\s*Amount\b\s*[:\-]?\s*({_NUM})",

            re.IGNORECASE,

        ),

        re.compile(

            r"(?:Net\s*(?:Bill\s*)?(?:Amount\s*)?Payable|Net\s*Payable|Net\s*Bill\s*Amount)"

            rf"\D{{0,20}}?({_NUM})",

            re.IGNORECASE,

        ),

        re.compile(

            rf"(?:Total\s*(?:Bill\s*)?Amount|Amount\s*Payable|Total\s*Payable|Current\s*Bill\s*Amount)"

            rf"\D{{0,20}}?({_NUM})",

            re.IGNORECASE,

        ),

    ],

}

# Bill-month patterns. Each one has a handler in ``_parse_month_match``.

_MONTH_NAMES = r"(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)[a-z]*"

_BILL_LABEL = r"(?:Bill(?:ing)?\s*(?:Month|Period|Date|For)|Month\s*of\s*Bill|Bill\s*Mth)"

_RE_MONTH_NAME_LABELLED = re.compile(

    rf"{_BILL_LABEL}\D{{0,12}}?{_MONTH_NAMES}[\s\-/,\.']*(\d{{4}}|\d{{2}})(?!\d)", re.IGNORECASE

)

_RE_MONTH_NAME_ANY = re.compile(

    rf"(?<![A-Za-z]){_MONTH_NAMES}[\s\-/,\.']*(\d{{4}})(?!\d)", re.IGNORECASE

)

_RE_NUMERIC_LABELLED = re.compile(

    rf"{_BILL_LABEL}\D{{0,12}}?(?:(\d{{1,2}})[\-/\.])?(\d{{1,2}})[\-/\.](\d{{4}}|\d{{2}})(?!\d)",

    re.IGNORECASE,

)

_RE_ISO_LABELLED = re.compile(

    rf"{_BILL_LABEL}\D{{0,12}}?(\d{{4}})[\-/](\d{{1,2}})(?!\d)", re.IGNORECASE

)

_RE_12_DIGITS = re.compile(r"(?<!\d)(\d{12})(?!\d)")

# Result key order (also the contract of parse_fields' return value).

_FIELD_KEYS: tuple[str, ...] = (

    "consumer_no", "units", "prev_reading", "curr_reading", "total_amount", "bill_month",

)

# --------------------------------------------------------------------------- #

# EasyOCR lazy singleton

# --------------------------------------------------------------------------- #

_reader: Optional["easyocr.Reader"] = None

_reader_lock = threading.Lock()

def _get_reader() -> "easyocr.Reader":

    """Return the process-wide EasyOCR reader, creating it on first use.

    Loading the detection/recognition models takes several seconds, so the

    reader is built once (thread-safe, double-checked locking) and reused.

    CPU mode is forced because the target demo machines have no CUDA GPU; if a

    GPU is available EasyOCR will still work, just without acceleration here.

    Returns:

        A ready-to-use ``easyocr.Reader`` for English.

    Raises:

        RuntimeError: If the models cannot be loaded or downloaded.

    """

    global _reader

    if _reader is None:

        with _reader_lock:

            if _reader is None:

                try:

                    logger.info("Initialising EasyOCR reader (English, CPU)...")

                    _reader = easyocr.Reader(["en"], gpu=False, verbose=False)

                except Exception as exc:  # download/IO/torch errors vary widely

                    raise RuntimeError(

                        "Failed to initialise EasyOCR. Pre-download the models "

                        "(needs internet once) or check the PyTorch install."

                    ) from exc

    return _reader

# --------------------------------------------------------------------------- #

# 1. Pre-processing

# --------------------------------------------------------------------------- #

def preprocess(img_bgr: np.ndarray) -> np.ndarray:

    """Clean a bill photo so that OCR sees crisp, high-contrast text.

    Steps (in order):

        1. BGR -> grayscale (a 2-D grayscale or 4-channel BGRA input is also

           accepted and handled gracefully).

        2. Upscale 1.5x with ``cv2.INTER_CUBIC`` (small digits become readable).

        3. ``cv2.fastNlMeansDenoising`` (h=15) to remove sensor/JPEG noise.

        4. CLAHE (clip 2.0, 8x8 tiles) to even out lighting and shadows.

        5. Gaussian adaptive threshold (binary, block size 31, C = 11).

    Args:

        img_bgr: Input image as a ``uint8`` NumPy array (H x W x 3, BGR).

    Returns:

        A single-channel ``uint8`` binary image (values 0 or 255).

    Raises:

        ValueError: If ``img_bgr`` is not a non-empty NumPy image array.

    """

    if not isinstance(img_bgr, np.ndarray) or img_bgr.size == 0:

        raise ValueError("preprocess() expects a non-empty NumPy image array.")

    try:

        # Step 1: grayscale (tolerate already-gray and BGRA inputs).

        if img_bgr.ndim == 2:

            gray = img_bgr

        elif img_bgr.ndim == 3 and img_bgr.shape[2] == 3:

            gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

        elif img_bgr.ndim == 3 and img_bgr.shape[2] == 4:

            gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGRA2GRAY)

        else:

            raise ValueError(f"Unsupported image shape: {img_bgr.shape}")

        if gray.dtype != np.uint8:  # OpenCV denoise/CLAHE need 8-bit input

            gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        # Step 2: 1.5x cubic upscale.

        gray = cv2.resize(gray, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)

        # Step 3: non-local-means denoising.

        gray = cv2.fastNlMeansDenoising(gray, None, h=15)

        # Step 4: local contrast enhancement.

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

        gray = clahe.apply(gray)

        # Step 5: Gaussian adaptive threshold -> binary image.

        return cv2.adaptiveThreshold(

            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 11

        )

    except cv2.error as exc:

        raise ValueError(f"OpenCV failed during pre-processing: {exc}") from exc

# --------------------------------------------------------------------------- #

# 2. OCR

# --------------------------------------------------------------------------- #

def run_ocr(img: np.ndarray, detections=None) -> list[tuple[str, float]]:

    """Run EasyOCR on an image and return text with confidence scores.

    The reader is created lazily and cached (see ``_get_reader``). EasyOCR

    returns boxes in arbitrary order, so results are re-sorted into natural

    reading order (top-to-bottom, then left-to-right within a text row). This

    keeps a label such as "Units Consumed" adjacent to its value, which the

    regex stage relies on.

    Args:

        img: Image to read (typically the output of :func:`preprocess`).

    Returns:

        A list of ``(extracted_text, confidence)`` tuples with confidence in

        ``[0.0, 1.0]``. Empty/whitespace-only detections are dropped.

    Raises:

        ValueError: If ``img`` is not a non-empty NumPy array.

        RuntimeError: If the OCR engine fails to initialise or to read.

    """

    if not isinstance(img, np.ndarray) or img.size == 0:

        raise ValueError("run_ocr() expects a non-empty NumPy image array.")

    reader = _get_reader()

    try:

        # Each detection is (bbox, text, confidence); bbox = 4 corner points.

        if detections is None:
            detections = reader.readtext(img)

    except Exception as exc:

        raise RuntimeError(f"EasyOCR failed while reading the image: {exc}") from exc

    items: list[tuple[float, float, float, str, float]] = []  # (cy, cx, h, text, conf)

    for det in detections:

        try:

            bbox, text, conf = det

            text = str(text).strip()

            if not text:

                continue

            pts = np.asarray(bbox, dtype=np.float32)

            cy, cx = float(pts[:, 1].mean()), float(pts[:, 0].mean())

            height = float(pts[:, 1].max() - pts[:, 1].min())

            items.append((cy, cx, max(height, 1.0), text, float(conf)))

        except (ValueError, TypeError, IndexError):

            logger.debug("Skipping malformed EasyOCR detection: %r", det)

    if not items:

        return []

    # Group boxes into text rows: a box joins the current row if its vertical

    # centre is within \~0.6 median text heights of the row's running mean.

    items.sort(key=lambda t: t[0])

    tolerance = 0.6 * float(np.median([t[2] for t in items]))

    rows: list[list[tuple[float, float, float, str, float]]] = [[items[0]]]

    for it in items[1:]:

        row_cy = float(np.mean([r[0] for r in rows[-1]]))

        if abs(it[0] - row_cy) <= tolerance:

            rows[-1].append(it)

        else:

            rows.append([it])

    ordered: list[tuple[str, float]] = []

    for row in rows:

        for _, _, _, text, conf in sorted(row, key=lambda t: t[1]):

            ordered.append((text, max(0.0, min(1.0, conf))))

    return ordered

# --------------------------------------------------------------------------- #

# 3. Field parsing

# --------------------------------------------------------------------------- #

def _build_text(ocr_lines: list[tuple[str, float]]) -> tuple[str, list[tuple[int, int, float]]]:

    """Join OCR lines into one searchable string and remember each line's span.

    Lines are separated by a single newline so regexes can bridge a label box

    and its value box.

    Args:

        ocr_lines: ``(text, confidence)`` tuples from :func:`run_ocr`.

    Returns:

        ``(full_text, spans)`` where each span is ``(start, end, confidence)``

        giving the character range of one line inside ``full_text``.

    """

    parts: list[str] = []

    spans: list[tuple[int, int, float]] = []

    pos = 0

    for item in ocr_lines:

        try:

            text, conf = item

        except (ValueError, TypeError):

            continue

        text = re.sub(r"[ \t\r\f\v]+", " ", str(text)).strip()

        if not text:

            continue

        try:

            conf_f = max(0.0, min(1.0, float(conf)))

        except (ValueError, TypeError):

            conf_f = 0.0

        parts.append(text)

        spans.append((pos, pos + len(text), conf_f))

        pos += len(text) + 1  # +1 for the "\n" separator

    return "\n".join(parts), spans

def _span_confidence(spans: list[tuple[int, int, float]], start: int, end: int) -> float:

    """Confidence of a matched value = the *weakest* OCR box it overlaps.

    Using the minimum is deliberately conservative: a number stitched together

    from two boxes is only as trustworthy as its shakiest piece.

    Args:

        spans: Line spans from :func:`_build_text`.

        start: Start offset of the matched value in the full text.

        end: End offset (exclusive) of the matched value.

    Returns:

        Minimum confidence of overlapping lines, or 0.0 if none overlap.

    """

    confs = [c for (s, e, c) in spans if s < end and e > start]

    return min(confs) if confs else 0.0

def _fix_digits(raw: str) -> str:

    """Map digit look-alikes (O->0, I/l/|->1) and strip separators."""

    table = str.maketrans({"O": "0", "o": "0", "I": "1", "l": "1", "|": "1"})

    return re.sub(r"[\s\-]", "", raw).translate(table)

def _to_int(raw: str) -> Optional[int]:

    """Convert a captured digit string to ``int`` (None on failure)."""

    try:

        return int(raw.replace(",", ""))

    except ValueError:

        return None

def _to_float(raw: str) -> Optional[float]:

    """Convert a captured amount (may contain Indian-style commas) to ``float``."""

    try:

        return float(raw.replace(",", ""))

    except ValueError:

        return None

def _best_match(

    text: str,

    spans: list[tuple[int, int, float]],

    patterns: list[re.Pattern[str]],

    convert: Callable[[str], Any],

    validate: Callable[[Any], bool],

) -> tuple[Any, float]:

    """Find the most trustworthy keyword-anchored match for one field.

    Every match of every pattern is converted and validated; the candidate with

    the highest OCR confidence wins (ties go to the earliest occurrence, and to

    earlier -- more specific -- patterns).

    Args:

        text: Full OCR text.

        spans: Line spans for confidence lookup.

        patterns: Compiled regexes whose group 1 is the raw value.

        convert: Function turning the raw string into a typed value or None.

        validate: Sanity check on the typed value.

    Returns:

        ``(value, confidence)`` or ``(None, 0.0)`` when nothing valid matched.

    """

    best_val: Any = None

    best_conf = -1.0

    for pattern in patterns:

        for m in pattern.finditer(text):

            try:

                value = convert(m.group(1))

            except Exception:  # defensive: never let one bad match abort parsing

                continue

            if value is None or not validate(value):

                continue

            conf = _span_confidence(spans, m.start(1), m.end(1))

            if conf > best_conf:

                best_val, best_conf = value, conf

    if best_val is None:

        return None, 0.0

    return best_val, round(best_conf, 4)

def _format_month(year: int, month: int) -> Optional[str]:

    """Format (year, month) using ``BILL_MONTH_FORMAT``; None if invalid."""

    if not (1 <= month <= 12 and 2000 <= year <= 2100):

        return None

    return datetime(year, month, 1).strftime(BILL_MONTH_FORMAT)

def _expand_year(y: str) -> int:

    """Turn a 2- or 4-digit year string into a 4-digit year."""

    yi = int(y)

    return yi + 2000 if yi < 100 else yi

def _extract_bill_month(

    text: str, spans: list[tuple[int, int, float]]

) -> tuple[Optional[str], float]:

    """Extract the billing month and normalise it to ``BILL_MONTH_FORMAT``.

    Tries, in order: labelled month-name ("Bill Month: Aug-2024"), labelled ISO

    ("2024-08"), labelled numeric ("Bill Date 15/08/2024" or "08/2024"), and

    finally any unlabelled "Mon YYYY" in the text (confidence x ``FALLBACK_PENALTY``).

    Args:

        text: Full OCR text.

        spans: Line spans for confidence lookup.

    Returns:

        ``(month_string, confidence)`` or ``(None, 0.0)``.

    """

    # MSEDCL heading: "Bill of Supply for the Month of - March 2020".
    # This must precede the generic fallback, which can otherwise pick
    # "February-2020" from the historical consumption chart.
    supply_heading = re.compile(
        r"Bill\s+of\s+Supply\s+for\s+(?:the\s+)?Month\s+of\s*[-:]?\s*"
        r"(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)[a-z]*"
        r"[\s\-/,\.']*(\d{4}|\d{2})(?!\d)", re.IGNORECASE
    )
    for m in supply_heading.finditer(text):
        mon = _MONTHS.get(m.group(1)[:3].lower())
        out = _format_month(_expand_year(m.group(2)), mon) if mon else None
        if out:
            return out, round(_span_confidence(spans, m.start(1), m.end(2)), 4)

    # 1) Labelled, month name.

    for m in _RE_MONTH_NAME_LABELLED.finditer(text):

        mon = _MONTHS.get(m.group(1)[:3].lower())

        out = _format_month(_expand_year(m.group(2)), mon) if mon else None

        if out:

            return out, round(_span_confidence(spans, m.start(1), m.end(2)), 4)

    # 2) Labelled, ISO "YYYY-MM".

    for m in _RE_ISO_LABELLED.finditer(text):

        out = _format_month(int(m.group(1)), int(m.group(2)))

        if out:

            return out, round(_span_confidence(spans, m.start(1), m.end(2)), 4)

    # 3) Labelled, numeric: "dd/mm/yyyy" or "mm/yyyy" (Indian day-first order).

    for m in _RE_NUMERIC_LABELLED.finditer(text):

        out = _format_month(_expand_year(m.group(3)), int(m.group(2)))

        if out:

            return out, round(_span_confidence(spans, m.start(2), m.end(3)), 4)

    # 4) Avoid unlabelled historical-chart months; they are not billing months.
    # Unlabelled dates are intentionally left for manual review.
    return None, 0.0

    # 4) Unlabelled "Aug 2024" anywhere (weaker evidence -> penalised).

    for m in _RE_MONTH_NAME_ANY.finditer(text):

        mon = _MONTHS.get(m.group(1)[:3].lower())

        out = _format_month(int(m.group(2)), mon) if mon else None

        if out:

            conf = _span_confidence(spans, m.start(1), m.end(2)) * FALLBACK_PENALTY

            return out, round(conf, 4)

    return None, 0.0

def parse_fields(ocr_lines: list[tuple[str, float]]) -> dict:

    """Extract the key MSEDCL bill fields from OCR output using regexes.

    Extraction is keyword-anchored rather than position-based, so it survives

    layout differences between bill versions. Per-field logic:

    * ``consumer_no``  -- 12 digits after "Consumer No/Number"; OCR look-alikes

      (O, I, l) are repaired. Fallback: a lone 12-digit number anywhere

      (confidence x ``FALLBACK_PENALTY``).

    * ``units``        -- "Units Consumed" / "Total Consumption". Fallback:

      ``curr_reading - prev_reading`` when both are known.

    * ``prev_reading`` / ``curr_reading`` -- "Previous/Present/Current Reading".

    * ``total_amount`` -- "Net Payable" / "Total Amount" / "Amount Payable";

      Indian comma grouping (1,23,456.00) is handled.

    * ``bill_month``   -- normalised to ``BILL_MONTH_FORMAT`` (default "YYYY-MM").

    Consistency check: if all three of units / prev / curr are found and

    ``curr - prev`` differs from ``units`` by more than 1, their confidences are

    capped at ``INCONSISTENT_CONF_CAP`` so the validation UI flags them.

    Args:

        ocr_lines: ``(text, confidence)`` tuples, as returned by :func:`run_ocr`.

    Returns:

        A dict with exactly the keys ``consumer_no``, ``units``,

        ``prev_reading``, ``curr_reading``, ``total_amount`` and ``bill_month``.

        Each value is a ``(value, confidence)`` tuple. When nothing matched the

        fallback is ``(None, 0.0)`` for strings and ``(0, 0.0)`` / ``(0.0, 0.0)``

        for numbers -- a confidence of 0.0 always means "not found".

    """

    # Safe defaults: every key is always present.

    result: dict[str, tuple[Any, float]] = {

        "consumer_no": (None, 0.0),

        "units": (0, 0.0),

        "prev_reading": (0, 0.0),

        "curr_reading": (0, 0.0),

        "total_amount": (0.0, 0.0),

        "bill_month": (None, 0.0),

    }

    try:

        if not ocr_lines:

            return result

        text, spans = _build_text(ocr_lines)

        if not text:

            return result

        # ---- consumer number -------------------------------------------------

        val, conf = _best_match(

            text, spans, _PATTERNS["consumer_no"],

            convert=lambda raw: (lambda d: d if len(d) == 12 and d.isdigit() else None)(_fix_digits(raw)),

            validate=lambda v: True,

        )

        if val is None:  # fallback: any isolated 12-digit number

            fallback_patterns = [_RE_12_DIGITS]

            val, conf = _best_match(

                text, spans, fallback_patterns, convert=str, validate=lambda v: True

            )

            conf = round(conf * FALLBACK_PENALTY, 4)

        if val is not None:

            result["consumer_no"] = (val, conf)

        # ---- integer fields --------------------------------------------------

        int_specs = {

            "units": lambda v: 0 <= v <= 99_999,

            "prev_reading": lambda v: 0 <= v <= 9_999_999,

            "curr_reading": lambda v: 0 <= v <= 9_999_999,

        }

        for key, check in int_specs.items():

            val, conf = _best_match(text, spans, _PATTERNS[key], _to_int, check)

            if val is not None:

                result[key] = (val, conf)

        # ---- total amount ----------------------------------------------------

        val, conf = _best_match(

            text, spans, _PATTERNS["total_amount"], _to_float, lambda v: 0 < v < 10_000_000

        )

        if val is not None:

            result["total_amount"] = (round(val, 2), conf)

        # ---- billing month ---------------------------------------------------

        val, conf = _extract_bill_month(text, spans)

        if val is not None:

            result["bill_month"] = (val, conf)

        # ---- cross-field logic: derive / validate units ----------------------

        units, units_c = result["units"]

        prev, prev_c = result["prev_reading"]

        curr, curr_c = result["curr_reading"]

        have_readings = prev_c > 0 and curr_c > 0 and curr >= prev

        if units_c == 0.0 and have_readings:

            # Units not found: derive from the two meter readings.

            derived = curr - prev

            if 0 <= derived <= 99_999:

                result["units"] = (derived, round(min(prev_c, curr_c) * FALLBACK_PENALTY, 4))

        elif units_c > 0.0 and have_readings and abs((curr - prev) - units) > 1:

            # Reading arithmetic fails: force human review of all three fields.

            cap = INCONSISTENT_CONF_CAP

            result["units"] = (units, min(units_c, cap))

            result["prev_reading"] = (prev, min(prev_c, cap))

            result["curr_reading"] = (curr, min(curr_c, cap))

            logger.warning(

                "Reading mismatch: curr(%s) - prev(%s) != units(%s)", curr, prev, units

            )

    except Exception:  # parsing must never crash the upload flow

        logger.exception("Unexpected error in parse_fields(); returning partial result")

    return result

# --------------------------------------------------------------------------- #

# 4. High-level façade

# --------------------------------------------------------------------------- #

def _limit_size(img: np.ndarray, max_side: int = MAX_INPUT_SIDE) -> np.ndarray:

    """Downscale (never upscale) an image so its longest side is <= ``max_side``."""

    h, w = img.shape[:2]

    longest = max(h, w)

    if longest <= max_side:

        return img

    scale = max_side / float(longest)

    return cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

def _extract_meter_table(img, detections=None):

    """Read MSEDCL meter table using OCR box positions."""

    reader = _get_reader()

    if detections is None:
        detections = reader.readtext(img)

    items = []

    for box, text, conf in detections:

        x = sum(p[0] for p in box) / 4

        y = sum(p[1] for p in box) / 4

        items.append((x, y, str(text).strip(), float(conf)))

    # Find the column headings.

    headings = {}

    for x, y, text, conf in items:

        label = text.lower().strip()

        if "units" in label or "unlts" in label:

            headings["units"] = (x, y)

        elif "previous" in label:

            headings["prev_reading"] = (x, y)

        elif "meter reading" in label or "meler reading" in label:

            # The first meter-reading heading is the current reading.

            if "curr_reading" not in headings:

                headings["curr_reading"] = (x, y)

    result = {}

    for field, (hx, hy) in headings.items():

        candidates = []

        for x, y, text, conf in items:

            if y <= hy + 5 or y > hy + 100:

                continue

            if not re.fullmatch(r"\d{1,7}", text):

                continue

            distance = abs(x - hx)

            if distance <= 45:

                candidates.append((y - hy, distance, int(text), conf))

        if candidates:

            candidates.sort(key=lambda item: (item[0], item[1]))

            _, _, value, confidence = candidates[0]

            result[field] = (value, round(confidence, 4))

    return result

def _extract_layout_dates(detections, image_shape):
    """Read the printed bill heading and due-date rows using OCR coordinates.

    Do not use historical charts, bill periods, or receipt/payment rows as dates.
    Return only supported candidates; the normal parser remains the fallback.
    """
    h, w = image_shape[:2]
    boxes = []
    for item in detections:
        try:
            pts, raw, confidence = item
            xs = [float(pt[0]) for pt in pts]
            ys = [float(pt[1]) for pt in pts]
            boxes.append((min(xs), max(xs), min(ys), max(ys),
                          str(raw).strip(), max(0., min(1., float(confidence)))))
        except (ValueError, TypeError, IndexError):
            continue
    boxes.sort(key=lambda b: (b[2], b[0]))
    found = {}
    month_re = re.compile(r"\b(Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\W*(20\d{2})\b", re.I)
    date_re = re.compile(r"(?<!\d)(\d{1,2})[./\-](\d{1,2})[./\-](20\d{2}|\d{2})(?!\d)")

    # The printed heading occupies the top 18% of the bill, not the chart.
    top = [b for b in boxes if (b[2] + b[3]) / 2 < h * .18]
    for b in top:
        m = month_re.search(b[4])
        if m:
            month = _MONTHS.get(m.group(1)[:3].lower())
            if month:
                found['bill_month'] = (_format_month(int(m.group(2)), month), round(b[5], 4))
                break
    if 'bill_month' not in found:
        # Text in heading can be split across separate boxes.
        heading = ' '.join(b[4] for b in top)
        m = re.search(r"(?:Bill\s+of\s+Supply|Month\s+of)[\s\S]{0,90}?" + month_re.pattern, heading, re.I)
        if m:
            month = _MONTHS.get(m.group(1)[:3].lower())
            if month:
                found['bill_month'] = (_format_month(int(m.group(2)), month), .55)

    # Due dates: only rows labelled 'Due Date' or 'Amount Due Date'.
    # Search rightwards on the same visual row (or next nearby row).
    matches = []
    for label in boxes:
        label_text = label[4]
        if not re.search(r"\b(?:Amount\s+)?Due\s+Date\b|\bDue\s+Date\b", label_text, re.I):
            continue
        if re.search(r"Last\s+Due|Previous|Receipt|Paid", label_text, re.I):
            continue
        ly = (label[2] + label[3]) / 2
        lh = max(label[3] - label[2], 8)
        candidates = [(label[4], label[5], 0.)]
        for b in boxes:
            cy = (b[2] + b[3]) / 2
            dy = abs(cy - ly)
            if b is label or dy > max(lh * 1.3, h * .012):
                continue
            if b[0] < label[0] or b[0] > label[1] + w * .30:
                continue
            candidates.append((b[4], b[5], dy))
        for raw, conf, dy in candidates:
            m = date_re.search(raw)
            if not m:
                continue
            try:
                date = datetime(_expand_year(m.group(3)), int(m.group(2)), int(m.group(1)))
            except ValueError:
                continue
            # Prefer the top-right prominent amount-due-date field.
            score = (2 if ly < h * .25 else 1) + (1 if 'Amount' in label_text else 0) - dy / max(h, 1)
            matches.append((score, date.strftime('%Y-%m-%d'), min(label[5], conf)))
    if matches:
        matches.sort(reverse=True)
        _, value, conf = matches[0]
        found['due_date'] = (value, round(conf, 4))
    return found


def extract_bill(image_bytes: bytes) -> dict:

    """One-call façade used by ``app.py``: raw upload bytes -> parsed bill dict.

    Flow: ``cv2.imdecode`` -> (optional downscale of huge photos) ->

    :func:`preprocess` -> :func:`run_ocr` -> :func:`parse_fields`.

    Args:

        image_bytes: Raw bytes of an uploaded JPG/PNG (e.g.

            ``st.file_uploader(...).getvalue()``).

    Returns:

        The dict from :func:`parse_fields`: keys ``consumer_no``, ``units``,

        ``prev_reading``, ``curr_reading``, ``total_amount``, ``bill_month``,

        each mapped to a ``(value, confidence)`` tuple.

    Raises:

        ValueError: If the bytes are empty or cannot be decoded as an image.

        RuntimeError: If the OCR engine fails (propagated from :func:`run_ocr`),

            in which case the UI should fall back to manual entry.

    """

    if not image_bytes:

        raise ValueError("No image data received (empty bytes).")

    # io.BytesIO gives a zero-copy view over bytes/bytearray/memoryview uploads.

    buffer = io.BytesIO(image_bytes).getbuffer()

    arr = np.frombuffer(buffer, dtype=np.uint8)

    img_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)

    if img_bgr is None:

        raise ValueError("Could not decode the file as an image (use JPG or PNG).")

    img_bgr = _limit_size(img_bgr)

    # Run the expensive EasyOCR detection/recognition once and reuse its boxes.
    try:
        detections = _get_reader().readtext(img_bgr)
    except Exception as exc:
        raise RuntimeError(f"EasyOCR failed while reading the image: {exc}") from exc

    lines = run_ocr(img_bgr, detections=detections)

    logger.info("OCR produced %d text boxes", len(lines))

    result = parse_fields(lines)

    # Prefer spatially anchored printed dates over unrelated historical dates.
    layout_dates = _extract_layout_dates(detections, img_bgr.shape)
    result.update(layout_dates)

    # Read meter table using OCR coordinates.

    table_fields = _extract_meter_table(img_bgr, detections=detections)

    for field, value in table_fields.items():

        if result[field][1] == 0.0:

            result[field] = value

    # Meter tables can contain a second "Meter Reading Type" column.
    # The plain text regex may mistake that type code (e.g. 177) for the
    # current meter reading (e.g. 3275). Prefer the position-based value
    # only when it agrees with independently detected previous/units.
    prev, prev_conf = result["prev_reading"]
    units, units_conf = result["units"]
    curr, curr_conf = result["curr_reading"]
    if prev_conf > 0 and units_conf > 0:
        expected = prev + units
        table_curr = table_fields.get("curr_reading")
        if (table_curr is not None and table_curr[1] > 0
                and abs(table_curr[0] - expected) <= 1):
            result["curr_reading"] = table_curr
        elif curr_conf == 0 or abs(curr - expected) > 1:
            # Derived suggestion, not an OCR reading; keep confidence low
            # so the review form can flag it for confirmation.
            if 0 <= expected <= 9_999_999:
                result["curr_reading"] = (
                    expected, round(min(prev_conf, units_conf) * FALLBACK_PENALTY, 4)
                )

    # Check that readings agree with units.

    prev, prev_conf = result["prev_reading"]

    curr, curr_conf = result["curr_reading"]

    units, units_conf = result["units"]

    if prev_conf > 0 and curr_conf > 0 and units_conf > 0:

        if abs((curr - prev) - units) > 1:

            for field in ("prev_reading", "curr_reading", "units"):

                value, confidence = result[field]

                result[field] = (

                    value,

                    min(confidence, INCONSISTENT_CONF_CAP),

                )

            logger.warning("Meter readings and units do not match.")

    return result

if __name__ == "__main__":  # quick manual test:  python ocr_engine.py bill.jpg

    import json

    import sys

    logging.basicConfig(level=logging.INFO)

    if len(sys.argv) != 2:

        print("Usage: python ocr_engine.py \<bill_image.jpg>")

        sys.exit(1)

    with open(sys.argv[1], "rb") as fh:

        print(json.dumps(extract_bill(fh.read()), indent=2, default=str))

