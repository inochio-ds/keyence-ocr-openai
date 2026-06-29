import json
import csv
import io
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment
from fastapi.responses import StreamingResponse
import logging
import os
import time
from email.header import decode_header
from datetime import datetime, timezone
from typing import Any, Dict, Optional, List, Tuple
import requests
import uvicorn
from dotenv import load_dotenv
import re
from fastapi import FastAPI, File, Query, Request, UploadFile
from fastapi.responses import JSONResponse, Response
from requests import Response as RequestsResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

load_dotenv()

APP_VERSION = "2026-06-26.read-only-optimized"


class Config:
    DOCINT_ENDPOINT = os.getenv("DOCINT_ENDPOINT", "").rstrip("/")
    DOCINT_KEY = os.getenv("DOCINT_KEY", "")
    DOCINT_API_VERSION = os.getenv("DOCINT_API_VERSION", "2024-11-30")

    # Cost-saving mode: OCR Read only.
    # Important: if Azure App Settings defines DOCINT_MODEL_ID, set it to prebuilt-read there too.
    DOCINT_MODEL_ID = os.getenv("DOCINT_MODEL_ID", "prebuilt-read")

    AOAI_ENDPOINT = os.getenv("AOAI_ENDPOINT", "").rstrip("/")
    AOAI_KEY = os.getenv("AOAI_KEY", "")
    AOAI_DEPLOYMENT = os.getenv("AOAI_DEPLOYMENT", "")
    AOAI_API_VERSION = os.getenv("AOAI_API_VERSION", "2024-02-01")

    HTTP_CONNECT_TIMEOUT_SEC = float(os.getenv("HTTP_CONNECT_TIMEOUT_SEC", "10"))
    HTTP_READ_TIMEOUT_SEC = float(os.getenv("HTTP_READ_TIMEOUT_SEC", "60"))
    DOCINT_POLL_MAX_RETRIES = int(os.getenv("DOCINT_POLL_MAX_RETRIES", "30"))
    DOCINT_POLL_INTERVAL_SEC = float(os.getenv("DOCINT_POLL_INTERVAL_SEC", "2"))
    DOCINT_POLL_BACKOFF_MULTIPLIER = float(os.getenv("DOCINT_POLL_BACKOFF_MULTIPLIER", "1.2"))
    DOCINT_POLL_MAX_INTERVAL_SEC = float(os.getenv("DOCINT_POLL_MAX_INTERVAL_SEC", "10"))


class APIError(Exception):
    def __init__(self, message: str, status_code: int = 500, details: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.details = details or {}


setup_logging_done = False


def setup_logging() -> None:
    global setup_logging_done
    if setup_logging_done:
        return
    log_level = os.getenv("LOG_LEVEL", "INFO").upper()
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    setup_logging_done = True


app = FastAPI(title="OCR + AI Extraction API", version=APP_VERSION)
setup_logging()


@app.exception_handler(APIError)
async def api_error_handler(_: Request, err: APIError) -> JSONResponse:
    payload: Dict[str, Any] = {"error": err.message}
    if err.details:
        payload["details"] = err.details
    logging.warning("API error: %s", payload)
    return JSONResponse(status_code=err.status_code, content=payload)


@app.exception_handler(requests.Timeout)
async def timeout_handler(_: Request, err: requests.Timeout) -> JSONResponse:
    logging.exception("Timeout while calling external service")
    return JSONResponse(status_code=504, content={"error": "External service timeout.", "details": {"reason": str(err)}})


@app.exception_handler(requests.RequestException)
async def request_exception_handler(_: Request, err: requests.RequestException) -> JSONResponse:
    logging.exception("Request exception while calling external service")
    return JSONResponse(status_code=502, content={"error": "External service request failed.", "details": {"reason": str(err)}})


@app.exception_handler(StarletteHTTPException)
async def starlette_http_exception_handler(_: Request, err: StarletteHTTPException) -> JSONResponse:
    return JSONResponse(status_code=err.status_code, content={"error": "HTTP error", "details": {"reason": str(err.detail)}})


@app.exception_handler(Exception)
async def unexpected_error_handler(_: Request, err: Exception) -> JSONResponse:
    logging.exception("Unexpected server error")
    return JSONResponse(status_code=500, content={"error": "Internal server error.", "details": {"reason": str(err)}})


@app.get("/health")
async def health() -> Dict[str, str]:
    return {
        "status": "ok",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "version": APP_VERSION,
    }


def _decode_header_value(raw: Optional[str]) -> str:
    if not raw:
        return ""
    try:
        return raw.encode("latin-1").decode("utf-8")
    except (UnicodeDecodeError, UnicodeEncodeError):
        return raw


@app.post("/process")
async def process_document(
    request: Request,
    file: UploadFile = File(...),
    prompt_file: Optional[UploadFile] = File(default=None),
    format: Optional[str] = Query(default=None),
    prompt: Optional[str] = Query(default=None),
    filename: Optional[str] = Query(default=None),
) -> Response:
    validate_required_env()

    if not file or not file.filename:
        raise APIError("No file provided.", 400)

    upload_filename = os.path.basename(file.filename)
    content_type = file.content_type or "application/octet-stream"

    if not is_allowed_file(upload_filename, content_type):
        raise APIError("Unsupported file type. Allowed: PDF or image formats.", 400)

    file_bytes = await file.read()
    if not file_bytes:
        raise APIError("Uploaded file is empty.", 400)

    txt_prompt = ""
    if prompt_file and prompt_file.filename:
        txt_ext = os.path.splitext(prompt_file.filename.lower())[1]
        if txt_ext != ".txt":
            raise APIError("prompt_file must be a .txt file.", 400)
        txt_bytes = await prompt_file.read()
        txt_prompt = txt_bytes.decode("utf-8", errors="replace").strip()

    x_format = _decode_header_value(request.headers.get("x-format"))
    output_filename = filename
    ext_format: Optional[str] = None
    if output_filename:
        ext = os.path.splitext(output_filename.lower())[1].lstrip(".")
        if ext in ("json", "csv", "xlsx", "txt"):
            ext_format = ext

    output_format = (format or ext_format or x_format or "xlsx").lower()
    user_prompt = (txt_prompt or prompt or "").strip()

    if not output_filename:
        output_filename = f"result.{output_format}"

    logging.info(
        "Processing request for file=%s, content_type=%s, format=%s, output_filename=%s, prompt_source=%s",
        upload_filename,
        content_type,
        output_format,
        output_filename,
        "txt_file" if txt_prompt else "query_param" if prompt else "none",
    )
    logging.info("[PROMPT] user_prompt: %s", {user_prompt})

    analyze_result = run_document_intelligence_analyze(file_bytes=file_bytes, content_type=content_type)
    ocr_text = (analyze_result.get("content") or "").strip()

    # GPT extracts all general fields from the full OCR text.
    ai_result = run_aoai_extraction(ocr_text=ocr_text, prompt=user_prompt)

    # Backend corrects only the special order-detail rows if detected.
    line_table_result = extract_order_detail_from_read_lines(analyze_result)
    if line_table_result:
        logging.info("Using optimized OCR Read line/table-band extraction and merging with GPT result")
        ai_result = remove_product_detail_keys(ai_result)
        ai_result.update(line_table_result)
    else:
        logging.info("No special order-detail table detected. Using GPT result only")

    decoded_filename = decode_mime_filename(file.filename)
    ai_result["ファイル参考"] = decoded_filename

    logging.info("DOCINT_MODEL_ID: %s", Config.DOCINT_MODEL_ID)
    logging.info("read_lines_count: %s", len(get_ocr_lines(analyze_result)))
    logging.info("ai_result: %s", ai_result)

    if output_format == "json":
        return Response(content=json.dumps(ai_result, ensure_ascii=False, indent=2), media_type="application/json; charset=utf-8")

    if output_format == "txt":
        lines = [f"{k}: {v}" for k, v in ai_result.items()]
        txt_content = "\n".join(lines)
        return Response(content=txt_content, media_type="text/plain; charset=utf-8", headers={"Content-Disposition": f"attachment; filename={output_filename}"})

    if output_format == "csv":
        data = flatten_json(ai_result)
        csv_text = "\ufeff" + dict_to_csv_vertical(data)
        return Response(content=csv_text, media_type="text/csv; charset=utf-8", headers={"Content-Disposition": f"attachment; filename={output_filename}"})

    xlsx_bytes = build_xlsx_from_ai_result(ai_result)
    return StreamingResponse(
        io.BytesIO(xlsx_bytes),
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": f"attachment; filename={output_filename}"},
    )


# -----------------------------
# Validation / file helpers
# -----------------------------

def validate_required_env() -> None:
    required = {
        "DOCINT_ENDPOINT": Config.DOCINT_ENDPOINT,
        "DOCINT_KEY": Config.DOCINT_KEY,
        "AOAI_ENDPOINT": Config.AOAI_ENDPOINT,
        "AOAI_KEY": Config.AOAI_KEY,
        "AOAI_DEPLOYMENT": Config.AOAI_DEPLOYMENT,
    }
    missing = [k for k, v in required.items() if not v]
    if missing:
        raise APIError("Missing required environment variables.", 500, {"missing": missing})


def decode_mime_filename(filename: str) -> str:
    try:
        decoded_parts = decode_header(filename)
        decoded_string = ""
        for part, encoding in decoded_parts:
            if isinstance(part, bytes):
                decoded_string += part.decode(encoding or "utf-8", errors="replace")
            else:
                decoded_string += part
        return decoded_string
    except Exception:
        return filename


def is_allowed_file(filename: str, content_type: str) -> bool:
    allowed_mime_prefixes = ["image/"]
    allowed_mimes = {"application/pdf"}
    allowed_ext = {".pdf", ".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".webp"}
    ext = os.path.splitext(filename.lower())[1]
    if ext in allowed_ext:
        return True
    if content_type in allowed_mimes:
        return True
    return any(content_type.startswith(prefix) for prefix in allowed_mime_prefixes)


# -----------------------------
# Azure Document Intelligence Read
# -----------------------------

def run_document_intelligence_analyze(file_bytes: bytes, content_type: str) -> dict:
    url = f"{Config.DOCINT_ENDPOINT}/documentintelligence/documentModels/{Config.DOCINT_MODEL_ID}:analyze?api-version={Config.DOCINT_API_VERSION}"
    headers = {"Ocp-Apim-Subscription-Key": Config.DOCINT_KEY, "Content-Type": content_type}

    logging.info("Submitting file to Azure Document Intelligence Read")
    logging.info("DOCINT_MODEL_ID: %s", Config.DOCINT_MODEL_ID)
    logging.info("DOCINT_API_VERSION: %s", Config.DOCINT_API_VERSION)

    response = requests.post(url, headers=headers, data=file_bytes, timeout=(Config.HTTP_CONNECT_TIMEOUT_SEC, Config.HTTP_READ_TIMEOUT_SEC))
    if response.status_code not in (200, 202):
        raise APIError("Document Intelligence analyze request failed.", 502, {"status_code": response.status_code, "response": safe_json_or_text(response)})

    operation_location = response.headers.get("Operation-Location")
    if not operation_location:
        payload = response.json() if response.content else {}
        return payload.get("analyzeResult") or {}

    return poll_docint_analyze_result(operation_location)


def poll_docint_analyze_result(operation_location: str) -> dict:
    headers = {"Ocp-Apim-Subscription-Key": Config.DOCINT_KEY}
    interval = Config.DOCINT_POLL_INTERVAL_SEC

    for attempt in range(1, Config.DOCINT_POLL_MAX_RETRIES + 1):
        logging.info("Polling OCR Read operation (attempt %s/%s)", attempt, Config.DOCINT_POLL_MAX_RETRIES)
        response = requests.get(operation_location, headers=headers, timeout=(Config.HTTP_CONNECT_TIMEOUT_SEC, Config.HTTP_READ_TIMEOUT_SEC))

        if response.status_code >= 400:
            raise APIError("Failed while polling Document Intelligence operation.", 502, {"status_code": response.status_code, "response": safe_json_or_text(response), "operation_location": operation_location})

        payload = response.json()
        status = str(payload.get("status", "")).lower()

        if status == "succeeded":
            return payload.get("analyzeResult") or {}

        if status in {"failed", "canceled", "cancelled"}:
            raise APIError("Document Intelligence operation did not succeed.", 502, {"status": status, "response": payload})

        if attempt < Config.DOCINT_POLL_MAX_RETRIES:
            time.sleep(interval)
            interval = min(interval * Config.DOCINT_POLL_BACKOFF_MULTIPLIER, Config.DOCINT_POLL_MAX_INTERVAL_SEC)

    raise APIError("Document Intelligence polling timed out before completion.", 504, {"max_retries": Config.DOCINT_POLL_MAX_RETRIES})


# -----------------------------
# Optimized OCR Read special order-detail parser
# -----------------------------

DIM_PATTERN = re.compile(r"\d+(?:\.\d+)?\s*[×xX]\s*\d+(?:\.\d+)?(?:\s*[×xX]\s*\d+(?:\.\d+)?\s*m?)?")
UNIT_WORDS = ["枚", "袋", "本", "個", "箱"]

# Tuning constants for READ line grouping.
# These are intentionally ratios/relative thresholds so they can work across PDF page sizes.
SAME_ROW_Y_RATIO = 0.012
BAND_END_MARGIN_RATIO = 0.004


def norm_text(value: str) -> str:
    if not value:
        return ""
    text = str(value)
    text = (
        text.replace("０", "0")
        .replace("１", "1")
        .replace("２", "2")
        .replace("３", "3")
        .replace("４", "4")
        .replace("５", "5")
        .replace("６", "6")
        .replace("７", "7")
        .replace("８", "8")
        .replace("９", "9")
        .replace("ｘ", "×")
        .replace("X", "×")
        .replace("x", "×")
        .replace("．", ".")
        .replace("，", ",")
        .strip()
    )
    text = re.sub(r"(\d)\.\s+(\d)", r"\1.\2", text)
    text = re.sub(r"(\d)\s+\.(\d)", r"\1.\2", text)
    text = re.sub(r"\s*[×xX]\s*", " × ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def split_name_spec(text: str) -> Tuple[str, str]:
    text = norm_text(text)
    m = DIM_PATTERN.search(text)
    if not m:
        return text, ""
    name = text[:m.start()].strip()
    spec = m.group(0).strip()
    name = re.sub(r"\s+0\.$", "", name).strip()
    spec = re.sub(r"^15\s*[×]", "0.15 ×", spec).strip()
    return name, spec


def get_polygon_xywh(polygon: Any) -> Tuple[float, float, float, float]:
    if not polygon:
        return 0.0, 0.0, 0.0, 0.0

    if isinstance(polygon, list) and polygon and isinstance(polygon[0], dict):
        xs = [float(p.get("x", 0)) for p in polygon]
        ys = [float(p.get("y", 0)) for p in polygon]
    elif isinstance(polygon, list) and len(polygon) >= 2:
        xs = [float(v) for v in polygon[0::2]]
        ys = [float(v) for v in polygon[1::2]]
    else:
        return 0.0, 0.0, 0.0, 0.0

    x0, x1 = min(xs), max(xs)
    y0, y1 = min(ys), max(ys)
    return x0, y0, x1 - x0, y1 - y0


def get_ocr_lines(analyze_result: dict) -> List[dict]:
    lines: List[dict] = []
    for page in analyze_result.get("pages", []):
        page_no = page.get("pageNumber", 0)
        page_w = float(page.get("width") or 1)
        page_h = float(page.get("height") or 1)
        unit = page.get("unit", "")
        for line in page.get("lines", []):
            text = norm_text(line.get("content", ""))
            if not text:
                continue
            x, y, w, h = get_polygon_xywh(line.get("polygon") or [])
            lines.append({
                "page": page_no,
                "page_w": page_w,
                "page_h": page_h,
                "unit": unit,
                "x": x,
                "y": y,
                "w": w,
                "h": h,
                "cx": x + w / 2,
                "cy": y + h / 2,
                "text": text,
            })
    lines.sort(key=lambda v: (v["page"], v["y"], v["x"]))
    return lines

def get_page_column_profile(page_lines: List[dict]) -> dict:
    profile = {
        "name_x": None,
        "qty_x": None,
        "unit_x": None,
        "price_x": None,
        "amount_x": None,
        "order_x": None,
        "due_x": None,
    }

    for line in page_lines:
        t = line["text"]

        if "品名" in t or "仕様" in t:
            profile["name_x"] = line["cx"]

        elif "荷姿数量" in t or t == "数量":
            profile["qty_x"] = line["cx"]

        elif "荷姿単位" in t or t == "単位":
            profile["unit_x"] = line["cx"]

        elif "単価" in t:
            profile["price_x"] = line["cx"]

        elif t == "金額":
            profile["amount_x"] = line["cx"]

        elif "客先注文番号" in t:
            profile["order_x"] = line["cx"]

        elif "回答納期" in t:
            profile["due_x"] = line["cx"]

    # fallback by x ratio if OCR header is missing
    if page_lines:
        page_w = page_lines[0].get("page_w") or 1

        profile["name_x"] = profile["name_x"] or page_w * 0.20
        profile["qty_x"] = profile["qty_x"] or page_w * 0.47
        profile["unit_x"] = profile["unit_x"] or page_w * 0.54
        profile["price_x"] = profile["price_x"] or page_w * 0.62
        profile["amount_x"] = profile["amount_x"] or page_w * 0.70
        profile["order_x"] = profile["order_x"] or page_w * 0.80
        profile["due_x"] = profile["due_x"] or page_w * 0.90

    return profile

def is_header_line(text: str) -> bool:
    header_keywords = ["品名", "仕様", "荷姿数量", "荷姿単位", "単価", "金額", "客先注文番号", "回答納期"]
    return any(k in text for k in header_keywords)

def is_noise_number(text: str) -> bool:
    t = norm_text(text)

    # mã fax / timestamp / id dài
    if re.fullmatch(r"\d{8,}", t):
        return True

    # timestamp-like
    if re.search(r"\d{4}/\d{2}/\d{2}", t):
        return True

    # page-like / scan code
    if re.search(r"P\.\d+|053225|000200", t):
        return True

    return False


def is_noise_or_footer(text: str) -> bool:
    noise_keywords = [
        "下請事業者殿",
        "支払条件",
        "合計金額",
        "回答納期欄",
        "000200-42",
        "P.001",
        "P.002",
        "P.003",
        "P.004",
        "▼",
        "注文書",
        "発注先",
        "扱い店",
        "納入先",
        "シーアイマテックス",
        "MATEX",
        "TEL",
        "FAX",
        "ページ",
        "日付",
        "受注No",
    ]
    return any(k in text for k in noise_keywords)


def contains_product_name(text: str) -> bool:
    return "加工品" in text


def contains_dimension(text: str) -> bool:
    return DIM_PATTERN.search(norm_text(text)) is not None


def has_special_order_detail_table(lines: List[dict]) -> bool:
    text_all = " ".join(l["text"] for l in lines)
    header_ok = all(k in text_all for k in ["品名", "仕様", "荷姿", "単価", "金額"])
    has_product_word = any(contains_product_name(l["text"]) for l in lines)
    has_dimension = any(contains_dimension(l["text"]) for l in lines)
    logging.info(
        "special_table_detection header_ok=%s has_product_word=%s has_dimension=%s",
        header_ok,
        has_product_word,
        has_dimension,
    )
    return header_ok and has_product_word and has_dimension


def is_product_start_line(line: dict, lines_same_page: List[dict]) -> bool:
    text = line["text"]
    if contains_product_name(text) and contains_dimension(text):
        return True

    if contains_product_name(text) and not contains_dimension(text):
        # Product name and dimension can be split across two nearby lines.
        page_h = line.get("page_h") or 1
        y_limit = line["y"] + page_h * 0.035
        for other in lines_same_page:
            if other["y"] <= line["y"]:
                continue
            if other["y"] > y_limit:
                break
            # Same left area, dimension-only line
            if contains_dimension(other["text"]):
                return True
    return False


def line_distance(a: dict, b: dict) -> float:
    return abs(a["cy"] - b["cy"]) + abs(a["cx"] - b["cx"]) * 0.15


def nearest_numeric_after_label(label_line: dict, band_lines: List[dict], used_ids: set) -> Optional[str]:
    candidates = []
    page_h = label_line.get("page_h") or 1
    page_w = label_line.get("page_w") or 1

    for idx, line in enumerate(band_lines):
        if idx in used_ids:
            continue

        text = line["text"].strip()

        # Skip parenthesized note values like "(68)"
        if re.fullmatch(r"\(\d+\)", text):
            continue

        # Only accept plain number
        if not re.fullmatch(r"\d+", text):
            continue

        # 原反 / 加工賃 should not be 0
        if text == "0":
            continue

        # Must be vertically close to label
        dy = abs(line["cy"] - label_line["cy"])
        if dy > page_h * 0.04:
            continue

        # Important:
        # Value should not be far to the left of label.
        # This prevents price-column numbers like 144400 from being captured as 加工賃.
        if line["cx"] < label_line["cx"] - page_w * 0.03:
            continue

        candidates.append((line_distance(label_line, line), idx, text))

    if not candidates:
        return None

    candidates.sort(key=lambda x: x[0])
    used_ids.add(candidates[0][1])
    return candidates[0][2]


def extract_number_from_same_or_next(label_regex: str, band_lines: List[dict], used_ids: set) -> str:
    # Wrap label_regex with non-capturing group.
    # Example:
    # label_regex = r"原\s*反|原区|原\s*区"
    # final regex = r"(?:原\s*反|原区|原\s*区)\s*(\(?\d+\)?)$"
    same_line_pattern = re.compile(rf"(?:{label_regex})\s*(\(?\d+\)?)$")
    label_only_pattern = re.compile(rf"^(?:{label_regex})$")

    # Case 1: label and number are in the same OCR line
    # Example:
    # 原反 24992
    # 加工賃 678
    # 原区 3185
    for idx, line in enumerate(band_lines):
        text = line["text"]

        m = same_line_pattern.search(text)
        if m and m.group(1):
            raw_value = m.group(1).strip()

            # Skip parenthesized note values like "(68)"
            if raw_value.startswith("(") and raw_value.endswith(")"):
                continue

            clean_value = raw_value.strip("()")

            # 原反 / 加工賃 should not be 0
            if clean_value == "0":
                continue

            used_ids.add(idx)
            return clean_value

    # Case 2: label-only line, number is near it
    # Example:
    # 原 反
    # 2849
    for idx, line in enumerate(band_lines):
        text = line["text"]

        if label_only_pattern.fullmatch(text):
            val = nearest_numeric_after_label(line, band_lines, used_ids)

            if val:
                # Extra safety
                if str(val).startswith("(") and str(val).endswith(")"):
                    continue

                if str(val) == "0":
                    continue

                used_ids.add(idx)
                return val

    return ""


def extract_order_number(band_lines: List[dict]) -> str:
    for line in band_lines:
        m = re.search(r"A\d+", line["text"])
        if m:
            return m.group(0)
    return ""


def extract_remark(band_lines: List[dict]) -> str:
    # Prefer meaningful text after 摘要 label, otherwise text containing 試験 or 様.
    for i, line in enumerate(band_lines):
        if line["text"] == "摘要":
            for nxt in band_lines[i + 1:i + 5]:
                t = nxt["text"]
                if t not in ["希望納期", "摘要"] and not re.fullmatch(r"\d+", t):
                    if not is_noise_or_footer(t):
                        return t
    for line in band_lines:
        t = line["text"]
        if ("試験" in t or "様" in t) and not is_noise_or_footer(t):
            return t
    return ""


def extract_quantity_and_unit(band_lines: List[dict], used_ids: set, profile: dict) -> Tuple[str, str]:
    qty_x = profile.get("qty_x")
    unit_x = profile.get("unit_x")

    unit_candidates = []

    for idx, line in enumerate(band_lines):
        t = line["text"]

        if t in UNIT_WORDS:
            unit_candidates.append((abs(line["cx"] - unit_x), idx, line))
        elif any(u in t for u in UNIT_WORDS) and not any(k in t for k in ["加工", "原", "反"]):
            unit_candidates.append((abs(line["cx"] - unit_x), idx, line))

    if not unit_candidates:
        return "", ""

    unit_candidates.sort(key=lambda x: x[0])
    _, unit_idx, unit_line = unit_candidates[0]

    unit = ""
    for u in UNIT_WORDS:
        if u in unit_line["text"]:
            unit = u
            break

    page_h = unit_line.get("page_h") or 1
    candidates = []

    for idx, line in enumerate(band_lines):
        if idx in used_ids:
            continue

        t = line["text"]

        if is_noise_number(t):
            continue

        if not re.fullmatch(r"\d+", t):
            continue

        try:
            n = int(t)
        except ValueError:
            continue

        # quantity should be small
        if not (1 <= n <= 99):
            continue

        # must be near quantity column
        if qty_x is not None:
            page_w = line.get("page_w") or 1
            if abs(line["cx"] - qty_x) > page_w * 0.08:
                continue

        # must be close vertically to unit
        dy = abs(line["cy"] - unit_line["cy"])
        if dy > page_h * 0.04:
            continue

        dist = abs(line["cx"] - qty_x) if qty_x is not None else line_distance(unit_line, line)
        candidates.append((dist, idx, t))

    qty = ""
    if candidates:
        candidates.sort(key=lambda x: x[0])
        _, qidx, qty = candidates[0]
        used_ids.add(qidx)

    used_ids.add(unit_idx)
    return qty, unit


def extract_price_amount(band_lines: List[dict], used_ids: set, profile: dict) -> Tuple[str, str]:
    price_x = profile.get("price_x")
    amount_x = profile.get("amount_x")
    order_x = profile.get("order_x")

    candidates = []

    for idx, line in enumerate(band_lines):
        if idx in used_ids:
            continue

        t = line["text"]

        if is_noise_number(t):
            continue

        if any(k in t for k in ["原", "反", "加工", "賃", "貸", "貨", "愛", "價"]):
            continue

        nums = re.findall(r"\d+", t)

        # Example: "28263 0"
        if len(nums) >= 2:
            price = nums[0]
            amount = nums[1]

            if len(price) < 4:
                continue

            if amount != "0":
                continue

            page_w = line.get("page_w") or 1

            # price line should be around price/amount columns, not far-right handwritten area
            if price_x is not None:
                if abs(line["cx"] - price_x) > page_w * 0.18:
                    continue

            # avoid lines near customer order/far right
            if order_x is not None and line["cx"] > order_x:
                continue

            candidates.append((abs(line["cx"] - price_x), line["y"], idx, price, amount))

    if candidates:
        candidates.sort(key=lambda x: (x[0], x[1]))
        _, _, idx, price, amount = candidates[0]
        used_ids.add(idx)
        return price, amount

    # fallback: standalone price
    standalone_prices = []

    for idx, line in enumerate(band_lines):
        if idx in used_ids:
            continue

        t = line["text"]

        if is_noise_number(t):
            continue

        if any(k in t for k in ["原", "反", "加工", "賃", "貸", "貨", "愛", "價"]):
            continue

        if not re.fullmatch(r"\d+", t):
            continue

        if len(t) < 4:
            continue

        page_w = line.get("page_w") or 1

        if price_x is not None:
            if abs(line["cx"] - price_x) > page_w * 0.12:
                continue

        if order_x is not None and line["cx"] > order_x:
            continue

        standalone_prices.append((abs(line["cx"] - price_x), line["y"], idx, t))

    if not standalone_prices:
        return "", ""

    standalone_prices.sort(key=lambda x: (x[0], x[1]))
    _, _, price_idx, price = standalone_prices[0]
    used_ids.add(price_idx)

    amount = ""
    price_line = band_lines[price_idx]
    page_h = price_line.get("page_h") or 1

    zero_candidates = []

    for j, zline in enumerate(band_lines):
        if j in used_ids:
            continue

        if zline["text"] != "0":
            continue

        if abs(zline["cy"] - price_line["cy"]) > page_h * 0.04:
            continue

        if amount_x is not None:
            zero_candidates.append((abs(zline["cx"] - amount_x), j))
        else:
            zero_candidates.append((line_distance(price_line, zline), j))

    if zero_candidates:
        zero_candidates.sort(key=lambda x: x[0])
        _, zero_idx = zero_candidates[0]
        amount = "0"
        used_ids.add(zero_idx)

    return price, amount

def resolve_product_name_from_band(product_line: dict, band_lines: List[dict]) -> str:
    text = product_line["text"].strip()

    # If product line already has full name, use it.
    if text != "加工品":
        return text

    # If OCR split product name as:
    # 白白コート5
    # 加工品
    # then look for the nearest previous text in the same band.
    candidates = []

    for line in band_lines:
        t = line["text"].strip()

        if line["y"] >= product_line["y"]:
            continue

        if not t:
            continue

        if contains_dimension(t):
            continue

        if is_header_line(t) or is_noise_or_footer(t):
            continue

        if re.fullmatch(r"\d+", t):
            continue

        if any(k in t for k in ["原", "反", "加工賃", "摘要", "希望納期"]):
            continue

        candidates.append((abs(product_line["y"] - line["y"]), t))

    if candidates:
        candidates.sort(key=lambda x: x[0])
        return f"{candidates[0][1]} 加工品"

    return text


def build_product_from_band(product_no: int, product_line: dict, band_lines: List[dict], next_line: Optional[dict], profile: dict) -> Dict[str, str]:
    used_ids: set = set()

    # Product name/spec can be same line or split to the next dimension line.
    if contains_product_name(product_line["text"]) and contains_dimension(product_line["text"]):
        name, spec = split_name_spec(product_line["text"])

        if name == "加工品":
            name = resolve_product_name_from_band(product_line, band_lines)

    else:
        name = resolve_product_name_from_band(product_line, band_lines)
        spec = ""
        for idx, line in enumerate(band_lines):
            if line is product_line:
                continue
            if contains_dimension(line["text"]):
                spec_match = DIM_PATTERN.search(line["text"])
                if spec_match:
                    spec = spec_match.group(0).strip()
                    used_ids.add(idx)
                    break

    order_no = extract_order_number(band_lines)
    remark = extract_remark(band_lines)

    # First extract quantity/unit and price/amount.
    # This prevents price values like 144400 from being captured as 加工賃.
    qty, unit = extract_quantity_and_unit(band_lines, used_ids, profile)
    price, amount = extract_price_amount(band_lines, used_ids, profile)

    # Then extract 原反 / 加工賃 from remaining unused numbers.
    gen = extract_number_from_same_or_next(r"原\s*反|原区|原\s*区", band_lines, used_ids)
    proc = extract_number_from_same_or_next(r"加工[賃貸貨愛價]?", band_lines, used_ids)

    # If amount is blank but OCR usually has zero amount, keep blank rather than guessing.
    return {
        f"品名_{product_no}": name,
        f"仕様_{product_no}": spec,
        f"荷姿数量_{product_no}": qty,
        f"荷姿単位_{product_no}": unit,
        f"単価_{product_no}": price,
        f"金額_{product_no}": amount,
        f"客先注文番号_{product_no}": order_no,
        f"回答納期_{product_no}": "",
        f"希望納期_{product_no}": "",
        f"摘要_{product_no}": remark,
        f"原反_{product_no}": gen,
        f"加工賃_{product_no}": proc,
    }


def extract_order_detail_from_read_lines(analyze_result: dict) -> dict:
    lines = get_ocr_lines(analyze_result)
    if not lines:
        return {}

    if not has_special_order_detail_table(lines):
        return {}

    result: Dict[str, Any] = {}
    product_no = 0

    # Process each page separately, preserving visual order.
    pages = sorted(set(l["page"] for l in lines))
    for page_no in pages:
        page_lines = [l for l in lines if l["page"] == page_no]
        if not page_lines:
            continue
        profile = get_page_column_profile(page_lines)
        logging.info("page=%s column_profile=%s", page_no, profile)

        product_starts = []
        for line in page_lines:
            if is_header_line(line["text"]) or is_noise_or_footer(line["text"]):
                continue
            if is_product_start_line(line, page_lines):
                product_starts.append(line)

        # Remove duplicate product-start detections that are too close vertically.
        deduped = []
        for line in product_starts:
            if not deduped:
                deduped.append(line)
                continue
            prev = deduped[-1]
            page_h = line.get("page_h") or 1
            if abs(line["y"] - prev["y"]) > page_h * 0.025:
                deduped.append(line)

        product_starts = deduped
        logging.info("page=%s product_start_count=%s", page_no, len(product_starts))

        for idx, start_line in enumerate(product_starts):
            next_start = product_starts[idx + 1] if idx + 1 < len(product_starts) else None
            page_h = start_line.get("page_h") or 1
            band_start_y = start_line["y"] - page_h * 0.004
            if next_start:
                band_end_y = next_start["y"] - page_h * BAND_END_MARGIN_RATIO
            else:
                # cuối trang: không cho band ăn xuống footer quá sâu
                footer_candidates = [
                    l for l in page_lines
                    if l["y"] > start_line["y"]
                    and any(k in l["text"] for k in ["合計金額", "下請事業者殿", "回答納期欄", "▼", "P."])
                ]

                if footer_candidates:
                    band_end_y = min(l["y"] for l in footer_candidates) - page_h * 0.004
                else:
                    band_end_y = start_line["y"] + page_h * 0.08

            band_lines = [
                l for l in page_lines
                if band_start_y <= l["y"] < band_end_y
                and not is_header_line(l["text"])
                and not is_noise_or_footer(l["text"])
            ]
            band_lines.sort(key=lambda v: (v["y"], v["x"]))

            # Guard: band must include a product name or dimension.
            if not any(contains_product_name(l["text"]) for l in band_lines):
                continue

            product_no += 1
            product_data = build_product_from_band(product_no, start_line, band_lines, next_start, profile)
            result.update(product_data)

    if product_no == 0:
        return {}

    logging.info("Extracted product detail count from OCR Read lines: %s", product_no)
    return result


def remove_product_detail_keys(data: dict) -> dict:
    product_key_pattern = re.compile(
        r"^(品名|仕様|荷姿数量|荷姿単位|単価|単価\(単位\)|金額|客先注文番号|回答納期|希望納期|摘要|原反|原区|加工賃|加工貸|加工貨|加工愛|加工價)_\d+$"
    )
    cleaned: Dict[str, Any] = {}
    for key, value in data.items():
        if isinstance(value, dict):
            cleaned[key] = remove_product_detail_keys(value)
            continue
        if product_key_pattern.match(str(key)):
            continue
        cleaned[key] = value
    return cleaned


# -----------------------------
# Azure OpenAI extraction
# -----------------------------

def run_aoai_extraction(ocr_text: str, prompt: str = "") -> Dict[str, Any]:
    url = f"{Config.AOAI_ENDPOINT}/openai/deployments/{Config.AOAI_DEPLOYMENT}/chat/completions?api-version={Config.AOAI_API_VERSION}"
    headers = {"api-key": Config.AOAI_KEY, "Content-Type": "application/json"}
    user_instruction = build_user_prompt(ocr_text=ocr_text, user_prompt=prompt)
    body = {
        "messages": [
            {"role": "system", "content": "Return JSON only"},
            {"role": "user", "content": user_instruction},
        ],
        "temperature": 0,
    }

    logging.info("Calling Azure OpenAI for structured extraction")
    response = requests.post(url, headers=headers, json=body, timeout=(Config.HTTP_CONNECT_TIMEOUT_SEC, Config.HTTP_READ_TIMEOUT_SEC))

    if response.status_code >= 400:
        raise APIError("Azure OpenAI request failed.", 502, {"status_code": response.status_code, "response": safe_json_or_text(response)})

    payload = response.json()
    content = extract_aoai_content(payload)
    parsed = parse_json_safely(content)
    if not isinstance(parsed, dict):
        raise APIError("AI output must be a JSON object.", 502, {"type": type(parsed).__name__})
    return parsed


def build_user_prompt(ocr_text: str, user_prompt: str) -> str:
    base = """
Extract structured data from the OCR text.
Rules:
- Keep values exactly as written in OCR
- JSON keys should be simple Japanese words
- Do not merge multiple labels into one field
- If unclear, return ""
- Return only valid JSON
- DO NOT put important fields into 補足 if they can be inferred
- Only use 補足 for truly irrelevant or unknown text
- Keep all values as-is
- Take all data OCR extracted
"""
    return base + "\nUser instructions:\n" + user_prompt + "\n\nOCR:\n" + ocr_text


# -----------------------------
# Output helpers
# -----------------------------

def dict_to_csv_vertical(data: Dict[str, Any]) -> str:
    output = io.StringIO()
    writer = csv.writer(output, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL, lineterminator="\r\n")
    for key, value in data.items():
        writer.writerow([key, value])
    return output.getvalue()


def extract_aoai_content(payload: Dict[str, Any]) -> str:
    try:
        return payload["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise APIError("Unexpected Azure OpenAI response shape.", 502, {"response": payload}) from exc


def flatten_json(data: Dict[str, Any], parent_key: str = "", sep: str = "_") -> Dict[str, Any]:
    items: Dict[str, Any] = {}
    for k, v in data.items():
        k = clean_key(str(k))
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.update(flatten_json(v, new_key, sep=sep))
        elif isinstance(v, list):
            items[new_key] = json.dumps(v, ensure_ascii=False)
        else:
            items[new_key] = v
    return items


def clean_key(key: str) -> str:
    return key.replace('"', "").replace("'", "").replace(" ", "").replace("\n", "").replace("\t", "").strip()


def split_summary_and_tables(data: dict, parent_key: str = ""):
    summary = {}
    tables = {}
    for k, v in data.items():
        key = f"{parent_key}_{k}" if parent_key else k
        if isinstance(v, dict):
            child_summary, child_tables = split_summary_and_tables(v, key)
            summary.update(child_summary)
            tables.update(child_tables)
        elif isinstance(v, list) and v and all(isinstance(i, dict) for i in v):
            tables[key] = v
        else:
            summary[key] = v
    return summary, tables


def write_vertical_summary(ws, summary: dict, start_row=1):
    row = start_row
    for k, v in summary.items():
        ws.cell(row=row, column=1, value=k)
        ws.cell(row=row, column=2, value=v)
        row += 1
    return row


def write_table(ws, rows: List[dict], start_row: int):
    if not rows:
        return start_row
    headers = list(rows[0].keys())
    special = ["原反", "加工賃"]
    normal_headers = [h for h in headers if h not in special]
    special_headers = [h for h in special if h in headers]
    headers = normal_headers + special_headers
    for col_idx, h in enumerate(headers, start=1):
        cell = ws.cell(row=start_row, column=col_idx, value=h)
        cell.font = Font(bold=True, color="FFFFFF")
        cell.fill = PatternFill("solid", fgColor="1F6D8C")
        cell.alignment = Alignment(horizontal="center")
    for r_idx, item in enumerate(rows, start=start_row + 1):
        for c_idx, h in enumerate(headers, start=1):
            value = item.get(h, "")
            if isinstance(value, list):
                value = ", ".join(str(v) for v in value)
            ws.cell(row=r_idx, column=c_idx, value=value)
    return start_row + len(rows) + 2


def build_xlsx_from_ai_result(ai_result: dict) -> bytes:
    wb = Workbook()
    ws = wb.active
    ws.title = "結果"
    summary, tables = split_summary_and_tables(ai_result)
    current_row = 1
    current_row = write_vertical_summary(ws, summary, start_row=current_row)
    current_row += 2
    for table_name, rows in tables.items():
        ws.cell(row=current_row, column=1, value=table_name)
        ws.cell(row=current_row, column=1).font = Font(bold=True)
        current_row += 1
        current_row = write_table(ws, rows, start_row=current_row)
    for col in ws.columns:
        max_length = 0
        col_letter = col[0].column_letter
        for cell in col:
            value = str(cell.value) if cell.value is not None else ""
            max_length = max(max_length, len(value))
        ws.column_dimensions[col_letter].width = min(max_length + 2, 40)
    output = io.BytesIO()
    wb.save(output)
    output.seek(0)
    return output.getvalue()


# -----------------------------
# JSON helpers
# -----------------------------

def parse_json_safely(content: str) -> Any:
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass
    cleaned = strip_markdown_fences(content)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    decoder = json.JSONDecoder()
    for i, ch in enumerate(cleaned):
        if ch not in "[{":
            continue
        try:
            obj, _ = decoder.raw_decode(cleaned[i:])
            return obj
        except json.JSONDecodeError:
            continue
    raise APIError("AI response was not valid JSON.", 502, {"raw": content[:2000]})


def strip_markdown_fences(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```") and stripped.endswith("```"):
        lines = stripped.splitlines()
        if len(lines) >= 3:
            return "\n".join(lines[1:-1]).strip()
    return stripped


def safe_json_or_text(response: RequestsResponse) -> Any:
    try:
        return response.json()
    except ValueError:
        return response.text[:4000]


if __name__ == "__main__":
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("app:app", host="127.0.0.1", port=port, reload=True, http="httptools")
