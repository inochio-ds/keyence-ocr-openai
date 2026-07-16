
import csv
import io
import json
import logging
import os
import re
import time
from datetime import datetime, timezone
from email.header import decode_header
from typing import Any, Dict, List, Optional, Tuple

import requests
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, File, Query, Request, UploadFile
from fastapi.responses import JSONResponse, Response, StreamingResponse
from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter
from requests import Response as RequestsResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

load_dotenv()
APP_VERSION = "2026-06-29.final-verified"


class Config:
    DOCINT_ENDPOINT = os.getenv("DOCINT_ENDPOINT", "").rstrip("/")
    DOCINT_KEY = os.getenv("DOCINT_KEY", "")
    DOCINT_API_VERSION = os.getenv("DOCINT_API_VERSION", "2024-11-30")
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


logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO").upper(), format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
app = FastAPI(title="OCR + AI Extraction API", version=APP_VERSION)


@app.exception_handler(APIError)
async def api_error_handler(_: Request, err: APIError) -> JSONResponse:
    payload = {"error": err.message}
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
    return {"status": "ok", "timestamp": datetime.now(timezone.utc).isoformat(), "version": APP_VERSION}


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
        if os.path.splitext(prompt_file.filename.lower())[1] != ".txt":
            raise APIError("prompt_file must be a .txt file.", 400)
        txt_prompt = (await prompt_file.read()).decode("utf-8", errors="replace").strip()

    x_format = _decode_header_value(request.headers.get("x-format"))
    output_filename = filename
    ext_format = None
    if output_filename:
        ext = os.path.splitext(output_filename.lower())[1].lstrip(".")
        if ext in ("json", "csv", "xlsx", "txt"):
            ext_format = ext
    output_format = (format or ext_format or x_format or "xlsx").lower()
    output_filename = output_filename or f"result.{output_format}"
    user_prompt = (txt_prompt or prompt or "").strip()

    logging.info(
        "Processing request for file=%s, content_type=%s, format=%s, output_filename=%s, prompt_source=%s",
        upload_filename, content_type, output_format, output_filename,
        "txt_file" if txt_prompt else "query_param" if prompt else "none",
    )
    # logging.info("[PROMPT] user_prompt: %s", {user_prompt})

    analyze_result = run_document_intelligence_analyze(file_bytes=file_bytes, content_type=content_type)
    ocr_text = (analyze_result.get("content") or "").strip()
    decoded_filename = decode_mime_filename(file.filename)
    logging.info(f"OCR chars = {ocr_text}")
    logging.info(f"OCR length = {len(ocr_text)} chars")
    

    # Always call GPT: backend handles table rows; GPT supplies non-table dynamic fields.
    ai_result = run_aoai_extraction(ocr_text=ocr_text, prompt=user_prompt)
    horizontal_rows = extract_order_rows_from_read_lines(analyze_result, decoded_filename)

    if horizontal_rows:
        horizontal_rows, extra_headers = enrich_order_rows_with_gpt_columns(horizontal_rows, ai_result)
        horizontal_rows = normalize_delivery_columns(horizontal_rows)

        # Do not output GPT-inferred 税額 unless the OCR actually contains a 税額 label.
        if "税額" in extra_headers and "税額" not in ocr_text:
            extra_headers.remove("税額")
            for row in horizontal_rows:
                row.pop("税額", None)

        row_dynamic_headers: List[str] = []
        for row in horizontal_rows:
            for key in row.keys():
                if key not in HORIZONTAL_HEADERS and key not in extra_headers and key not in row_dynamic_headers:
                    row_dynamic_headers.append(key)
        output_headers = HORIZONTAL_HEADERS + [h for h in extra_headers if h not in HORIZONTAL_HEADERS] + row_dynamic_headers
        logging.info("Using horizontal order-detail rows with normalized GPT extra columns. rows=%s extra_cols=%s", len(horizontal_rows), len(extra_headers))
        logging.info("extra_headers=%s", extra_headers)
        logging.info("output_headers=%s", output_headers)
        return build_response(horizontal_rows, output_headers, output_format, output_filename, sheet_name="注文明細")

    logging.info("No order-detail table detected. Using GPT dynamic horizontal rows")
    generic_rows, generic_headers = build_generic_dynamic_horizontal(ai_result, decoded_filename)
    return build_response(generic_rows, generic_headers, output_format, output_filename, sheet_name="抽出結果")


# -----------------------------
# Common helpers
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


def _decode_header_value(raw: Optional[str]) -> str:
    if not raw:
        return ""
    try:
        return raw.encode("latin-1").decode("utf-8")
    except Exception:
        return raw


def decode_mime_filename(filename: str) -> str:
    try:
        decoded = ""
        for part, encoding in decode_header(filename):
            decoded += part.decode(encoding or "utf-8", errors="replace") if isinstance(part, bytes) else part
        return decoded
    except Exception:
        return filename


def is_allowed_file(filename: str, content_type: str) -> bool:
    allowed_ext = {".pdf", ".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".webp"}
    return os.path.splitext(filename.lower())[1] in allowed_ext or content_type == "application/pdf" or content_type.startswith("image/")


def safe_json_or_text(response: RequestsResponse) -> Any:
    try:
        return response.json()
    except ValueError:
        return response.text[:4000]


def normalize_number_text(value: str) -> str:
    return (
        str(value)
        .replace("０", "0").replace("１", "1").replace("２", "2").replace("３", "3").replace("４", "4")
        .replace("５", "5").replace("６", "6").replace("７", "7").replace("８", "8").replace("９", "9")
        .replace("ー", "-").replace("−", "-").replace("－", "-").replace("‐", "-").replace("–", "-").replace("—", "-")
    )


# -----------------------------
# Document Intelligence
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
            raise APIError("Failed while polling Document Intelligence operation.", 502, {"status_code": response.status_code, "response": safe_json_or_text(response)})
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
# OCR table parser settings
# -----------------------------

HORIZONTAL_HEADERS = [
    "注文番号", "注文内明細番号", "発注No", "受注No", "ページ", "扱い店", "納入先", "納入先郵便番号", "納入先住所", "発注者", "日付",
    "発行会社", "発行元郵便番号", "発行元住所", "発行元TEL", "発行元FAX", "発行部署",
    "品名", "仕様", "荷姿数量", "荷姿単位", "単価", "金額", "客先注文番号", "回答納期", "希望納期",
    "摘要", "原反", "加工賃", "ファイル参考",
]

DIM_PATTERN = re.compile(r"\d+(?:\.\d+)?\s*[×xX]\s*\d+(?:\.\d+)?(?:\s*[×xX]\s*\d+(?:\.\d+)?\s*m?)?")
UNIT_WORDS = ["枚", "袋", "本", "個", "箱"]
BAND_END_MARGIN_RATIO = 0.004


def norm_text(value: str) -> str:
    if not value:
        return ""
    text = normalize_number_text(str(value))
    text = text.replace("ｘ", "×").replace("X", "×").replace("x", "×").replace("．", ".").replace("，", ",").strip()
    text = re.sub(r"(\d)\.\s+(\d)", r"\1.\2", text)
    text = re.sub(r"(\d)\s+\.(\d)", r"\1.\2", text)
    text = re.sub(r"\s*[×xX]\s*", " × ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


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
        for line in page.get("lines", []):
            text = norm_text(line.get("content", ""))
            if not text:
                continue
            x, y, w, h = get_polygon_xywh(line.get("polygon") or [])
            lines.append({"page": page_no, "page_w": page_w, "page_h": page_h, "x": x, "y": y, "w": w, "h": h, "cx": x + w / 2, "cy": y + h / 2, "text": text})
    lines.sort(key=lambda v: (v["page"], v["y"], v["x"]))
    return lines


def page_text(page_lines: List[dict]) -> str:
    return " ".join(l["text"] for l in sorted(page_lines, key=lambda x: (x["y"], x["x"])))


def extract_document_page_label(page_lines: List[dict]) -> str:
    """OCR printed page label, e.g. 1/2, 2/2, 1/1."""
    lines_sorted = sorted(page_lines, key=lambda x: (x["y"], x["x"]))
    if not lines_sorted:
        return ""
    page_w = lines_sorted[0].get("page_w", 1) or 1
    page_h = lines_sorted[0].get("page_h", 1) or 1
    text_all = " ".join(l["text"] for l in lines_sorted)
    m = re.search(r"ページ\s*[:：]?\s*(\d+)\s*/\s*(\d+)", text_all)
    if m:
        return f"{m.group(1)}/{m.group(2)}"
    for label in [l for l in lines_sorted if "ページ" in l["text"]]:
        nearby = [l["text"] for l in lines_sorted if abs(l["cy"] - label["cy"]) <= page_h * 0.045 and l["x"] >= label["x"] - page_w * 0.03]
        nearby_text = " ".join(nearby)
        m = re.search(r"(\d+)\s*/\s*(\d+)", nearby_text)
        if m:
            return f"{m.group(1)}/{m.group(2)}"
        nums = re.findall(r"\d+", nearby_text)
        if len(nums) >= 2:
            return f"{nums[0]}/{nums[1]}"
    return ""


def extract_page_header(page_lines: List[dict]) -> Dict[str, str]:
    lines_sorted = sorted(page_lines, key=lambda x: (x["y"], x["x"]))
    texts = [l["text"] for l in lines_sorted]
    text_all = " ".join(texts)

    header = {
        "発注No": "",
        "受注No": "",
        "日付": "",
        "扱い店": "",
        "納入先": "",
        "納入先郵便番号": "",
        "納入先住所": "",
        "発注者": "",
    }

    # ------------------------------------------------------------
    # Helper: extract No value near label.
    # Handles:
    #   受注No 22J0320xxx
    #   受注NO 22J0320xxx
    #   受注No. 22J0320xxx
    #   label and value split into nearby OCR lines
    # ------------------------------------------------------------
    def find_no_value_near_label(label_regex: str, prefer_prefix: str = "") -> str:
        # 1. Global same-text search
        m = re.search(
            rf"{label_regex}\s*\.?\s*[:：]?\s*([A-Za-z0-9]{{6,}})",
            text_all,
            flags=re.IGNORECASE,
        )
        if m:
            candidate = m.group(1).strip()
            if not prefer_prefix or candidate.startswith(prefer_prefix):
                return candidate

        # 2. Line-by-line same line search
        for i, line in enumerate(lines_sorted):
            t = line["text"]

            if not re.search(label_regex, t, flags=re.IGNORECASE):
                continue

            m = re.search(
                rf"{label_regex}\s*\.?\s*[:：]?\s*([A-Za-z0-9]{{6,}})",
                t,
                flags=re.IGNORECASE,
            )
            if m:
                candidate = m.group(1).strip()
                if not prefer_prefix or candidate.startswith(prefer_prefix):
                    return candidate

            # 3. Same horizontal band, right side of label
            page_w = line.get("page_w", 1) or 1
            page_h = line.get("page_h", 1) or 1
            nearby_candidates = []

            for other in lines_sorted:
                if other is line:
                    continue

                # same y band
                if abs(other["cy"] - line["cy"]) > page_h * 0.035:
                    continue

                # value is usually to the right or near label
                if other["x"] < line["x"] - page_w * 0.02:
                    continue

                ot = other["text"].strip()

                for candidate in re.findall(r"[A-Za-z0-9]{6,}", ot):
                    # Avoid taking 発注No as 受注No accidentally
                    if prefer_prefix and not candidate.startswith(prefer_prefix):
                        continue
                    nearby_candidates.append((abs(other["x"] - line["x"]), candidate))

            if nearby_candidates:
                nearby_candidates.sort(key=lambda x: x[0])
                return nearby_candidates[0][1]

            # 4. Next few OCR lines fallback
            for nxt in texts[i + 1:i + 5]:
                for candidate in re.findall(r"[A-Za-z0-9]{6,}", nxt):
                    if prefer_prefix and not candidate.startswith(prefer_prefix):
                        continue
                    return candidate

        return ""

    # ------------------------------------------------------------
    # 発注No / 受注No / 日付
    # ------------------------------------------------------------
    header["発注No"] = find_no_value_near_label(
        r"発\s*注\s*N[oO0０]?",
        prefer_prefix="22H",
    )

    header["受注No"] = find_no_value_near_label(
        r"受\s*注\s*N[oO0０]?",
        prefer_prefix="22J",
    )

    m = re.search(
        r"日付\s*[:：]?\s*([0-9]{4}年[0-9]{1,2}月[0-9]{1,2}日)",
        text_all,
    )
    if m:
        header["日付"] = m.group(1)

    # ------------------------------------------------------------
    # 発注者
    # ------------------------------------------------------------
    m = re.search(r"発注者\s+([^\s]+(?:\s+[^\s]+)?)", text_all)
    if m:
        header["発注者"] = m.group(1).strip()

    # ------------------------------------------------------------
    # 扱い店 / 納入先
    # ------------------------------------------------------------
    for t in texts:
        if "扱い店" in t and "納入先" in t:
            m = re.search(r"扱い店\s*[:：]?\s*(.*?)\s+納入先\s*[:：]?\s*(.*)", t)
            if m:
                header["扱い店"] = m.group(1).strip()
                delivery = m.group(2).strip()

                if delivery and not delivery.startswith("〒"):
                    header["納入先"] = delivery
            continue

        if "扱い店" in t and not header["扱い店"]:
            m = re.search(r"扱い店\s*[:：]?\s*(.*)", t)
            if m:
                header["扱い店"] = m.group(1).strip()

        if "納入先" in t and not header["納入先"]:
            m = re.search(r"納入先\s*[:：]?\s*(.*)", t)
            if m:
                delivery = m.group(1).strip()
                if delivery and not delivery.startswith("〒"):
                    header["納入先"] = delivery

    # ------------------------------------------------------------
    # Recover delivery name if attached to dealer line.
    # Example:
    #   宮田物産株式会社 ちのみ農園 知野見洋幸様
    # ------------------------------------------------------------
    dealer = header.get("扱い店", "")

    if dealer:
        for t in texts:
            if dealer in t and "様" in t:
                tail = t.split(dealer, 1)[1].strip()
                tail = re.sub(r"^(扱い店|納入先)\s*[:：]?", "", tail).strip()

                if tail and not tail.startswith("〒"):
                    if not header["納入先"] or header["納入先"].startswith("〒"):
                        header["納入先"] = tail
                break

    # If dealer itself still contains delivery name, split it.
    if header["扱い店"] and "様" in header["扱い店"]:
        parts = header["扱い店"].split()

        if len(parts) >= 2:
            header["扱い店"] = parts[0]
            delivery_name = " ".join(parts[1:]).strip()

            if delivery_name and (not header["納入先"] or header["納入先"].startswith("〒")):
                header["納入先"] = delivery_name
        else:
            header["扱い店"] = header["扱い店"].split()[0]

    # 納入先 should never be only a postal code.
    if str(header.get("納入先", "")).startswith("〒"):
        header["納入先"] = ""

    return header


def extract_issuer_info_from_page(page_lines: List[dict]) -> Dict[str, str]:
    lines_sorted = sorted(page_lines, key=lambda x: (x["y"], x["x"]))
    text_all = " ".join(l["text"] for l in lines_sorted)
    info = {"発行会社": "", "発行元郵便番号": "", "発行元住所": "", "発行元TEL": "", "発行元FAX": "", "発行部署": ""}
    page_w = lines_sorted[0].get("page_w", 1) if lines_sorted else 1

    company_line = None
    for line in lines_sorted:
        t = line["text"]
        if "アイマテックス" in t or "シーアイマテックス" in t:
            company_line = line
            m = re.search(r"([ァ-ヶ一-龥ーA-Za-z]*アイマテックス株式会社)", t)
            info["発行会社"] = m.group(1) if m else re.sub(r"^[A-Za-z0-9●&®も×\s]+", "", t).strip()
            break

    company_x = company_line["x"] if company_line else None
    postal_line = None
    candidates = []
    for line in lines_sorted:
        m = re.search(r"〒\s*\d{3}-\d{4}", line["text"])
        if m:
            candidates.append((abs(line["x"] - company_x) if company_x is not None else line["y"], line, m.group(0).replace(" ", "")))
    if candidates:
        candidates.sort(key=lambda x: x[0])
        _, postal_line, postal = candidates[0]
        info["発行元郵便番号"] = postal

    if postal_line:
        parts = []
        postal_x = postal_line["x"]
        for line in lines_sorted:
            if line["y"] <= postal_line["y"]:
                continue
            t = line["text"]
            if any(stop in t for stop in ["TEL", "FAX", "農業資材部", "発注者", "品名", "仕様", "注文書"]):
                break
            if t.startswith("〒"):
                continue
            if abs(line["x"] - postal_x) <= page_w * 0.12:
                parts.append(t)
        info["発行元住所"] = " ".join(parts).strip()

    phone_pattern = r"0[0-9０-９]{1,4}[\-ー−－‐–—][0-9０-９]{2,4}[\-ー−－‐–—][0-9０-９]{3,4}"
    phones = [normalize_number_text(p) for p in re.findall(phone_pattern, text_all)]
    for pat in [rf"TEL\s*[:：]?\s*({phone_pattern})", rf"T\s*E\s*L\s*[:：]?\s*({phone_pattern})"]:
        m = re.search(pat, text_all)
        if m:
            info["発行元TEL"] = normalize_number_text(m.group(1))
            break
    for pat in [rf"FAX\s*[:：]?\s*({phone_pattern})", rf"F\s*A\s*X\s*[:：]?\s*({phone_pattern})"]:
        m = re.search(pat, text_all)
        if m:
            info["発行元FAX"] = normalize_number_text(m.group(1))
            break
    if not info["発行元TEL"] and len(phones) >= 1:
        info["発行元TEL"] = phones[0]
    if not info["発行元FAX"] and len(phones) >= 2:
        info["発行元FAX"] = phones[1]

    for line in lines_sorted:
        if "農業資材部" in line["text"]:
            info["発行部署"] = line["text"].strip()
            break
    return info

def should_skip_gpt_extra_header(base_key: str) -> bool:
    key = str(base_key).strip().replace(" ", "")
    key = key.rstrip(".。")

    # GPT often returns duplicated order-level fields under 注文書_*.
    # These duplicate fixed columns: 発注No, 受注No, ページ, 日付.
    if key == "注文書":
        return True

    if key.startswith("注文書_"):
        return True

    return False


def extract_delivery_info_from_page(page_lines: List[dict]) -> Dict[str, str]:
    """
    Extract 納入先 / 納入先郵便番号 / 納入先住所 from the delivery block.

    This version is intentionally conservative for address extraction and more tolerant for name extraction:
    - postal/address are detected around the 納入先 label and never allow issuer postal 〒105-0014.
    - address stops before date/table/body text.
    - if the name is not between label and postal, search the area just above the postal line and recover
      names like "ちのみ農園 知野見洋幸様", including cases where it is attached to the dealer line.
    """
    lines_sorted = sorted(page_lines, key=lambda x: (x["y"], x["x"]))
    result = {"納入先": "", "納入先郵便番号": "", "納入先住所": ""}
    if not lines_sorted:
        return result

    page_w = lines_sorted[0].get("page_w", 1) or 1
    page_h = lines_sorted[0].get("page_h", 1) or 1

    label_lines = [l for l in lines_sorted if "納入先" in l["text"]]
    if not label_lines:
        return result
    label = label_lines[0]

    # 1) Find delivery postal code below the 納入先 label. Exclude issuer postal code.
    postal_candidates = []
    for line in lines_sorted:
        t = line["text"].strip()
        if not t:
            continue
        if line["y"] <= label["y"]:
            continue
        if line["y"] - label["y"] > page_h * 0.22:
            continue
        m = re.search(r"〒\s*\d{3}-\d{4}", t)
        if not m:
            continue
        postal = m.group(0).replace(" ", "")
        if postal == "〒105-0014":
            continue
        if abs(line["x"] - label["x"]) <= page_w * 0.70:
            postal_candidates.append((line["y"], abs(line["x"] - label["x"]), line, postal))

    if not postal_candidates:
        return result

    postal_candidates.sort(key=lambda x: (x[0], x[1]))
    _, _, postal_line, postal = postal_candidates[0]
    result["納入先郵便番号"] = postal

    def clean_delivery_name(raw: str) -> str:
        t = str(raw or "").strip()
        t = re.sub(r"^(扱い店|納入先)\s*[:：]?", "", t).strip()
        if not t or t.startswith("〒"):
            return ""
        # Remove dealer/company prefix when OCR merges dealer + delivery name.
        # Example: 宮田物産株式会社 ちのみ農園 知野見洋幸様 -> ちのみ農園 知野見洋幸様
        parts = t.split()
        if len(parts) >= 2 and ("株式会社" in parts[0] or parts[0].endswith("会社")):
            t = " ".join(parts[1:]).strip()
        # If the remaining string is only a company/dealer name, do not use it as delivery.
        if t.endswith("株式会社") and "様" not in t:
            return ""
        return t

    def is_bad_name_candidate(t: str) -> bool:
        if not t or t.startswith("〒"):
            return True
        if re.fullmatch(r"\d+", t):
            return True
        if re.search(r"\d{4}/\d{2}/\d{2}", t):
            return True
        if any(k in t for k in [
            "シーアイマテックス", "アイマテックス", "TEL", "FAX", "農業資材部", "発注者",
            "注文書", "ページ", "日付", "受注No", "発注No", "納入先", "下記", "数量", "単位",
            "品名", "仕様", "荷姿", "単価", "金額"
        ]):
            return True
        return False

    # 2) Primary name: any valid text between label and postal.
    name_candidates = []
    for line in lines_sorted:
        if line["y"] <= label["y"] or line["y"] >= postal_line["y"]:
            continue
        t = line["text"].strip()
        if is_bad_name_candidate(t):
            continue
        if abs(line["x"] - label["x"]) <= page_w * 0.70:
            cleaned = clean_delivery_name(t)
            if cleaned:
                name_candidates.append((line["y"], line["x"], cleaned))

    # 3) Fallback name: search just above the postal line, including slightly above the label.
    # This catches order 3 when the name is OCR-positioned as part of the dealer line.
    if not name_candidates:
        y_min = label["y"] - page_h * 0.10
        y_max = postal_line["y"] + page_h * 0.04
        for line in lines_sorted:
            t = line["text"].strip()
            if is_bad_name_candidate(t):
                continue
            if line["y"] < y_min or line["y"] > y_max:
                continue
            # Prefer lines containing 様, but also allow non-company text.
            cleaned = clean_delivery_name(t)
            if not cleaned:
                continue
            if "様" in cleaned or ("株式会社" not in cleaned and not cleaned.endswith("会社")):
                name_candidates.append((line["y"], line["x"], cleaned))

    if name_candidates:
        # Prefer candidates containing 様, then closest to postal line from above.
        name_candidates.sort(key=lambda x: (0 if "様" in x[2] else 1, abs(postal_line["y"] - x[0]), x[1]))
        result["納入先"] = name_candidates[0][2]

    # 4) Address: same line after postal + immediate following address line(s). Stop before date/table text.
    address_parts = []
    t = postal_line["text"].strip()
    m = re.search(r"〒\s*\d{3}-\d{4}", t)
    if m:
        after = t[m.end():].strip()
        if after:
            address_parts.append(after)

    for line in lines_sorted:
        if line["y"] <= postal_line["y"]:
            continue
        if line["y"] - postal_line["y"] > page_h * 0.07:
            continue
        t = line["text"].strip()
        if not t or t.startswith("〒"):
            continue
        if re.search(r"\d{4}/\d{2}/\d{2}", t):
            break
        if any(k in t for k in [
            "TEL", "FAX", "下記", "数量", "単位", "品名", "仕様", "荷姿", "単価", "金額",
            "注文書", "ページ", "日付", "受注No", "発注No"
        ]):
            break
        if abs(line["x"] - postal_line["x"]) <= page_w * 0.35:
            address_parts.append(t)

    result["納入先住所"] = " ".join(address_parts).strip()
    return result

def build_order_groups(lines: List[dict]) -> Tuple[List[dict], Dict[int, dict]]:
    pages = sorted(set(l["page"] for l in lines))
    orders: List[dict] = []
    current: Optional[dict] = None
    for page_no in pages:
        page_lines = [l for l in lines if l["page"] == page_no]
        header = extract_page_header(page_lines)
        issuer_info = extract_issuer_info_from_page(page_lines)
        delivery_info = extract_delivery_info_from_page(page_lines)

        # Keep header delivery name if found from dealer split; fill postal/address from position parser.
        for k, v in delivery_info.items():
            if v and (k != "納入先" or not header.get("納入先")):
                header[k] = v

        text_all = page_text(page_lines)
        has_delivery = bool(header.get("納入先")) or bool(header.get("納入先郵便番号")) or "納入先" in text_all
        different_order = bool(current and header.get("発注No") and current.get("発注No") and header["発注No"] != current["発注No"])
        starts_new = current is None or has_delivery or different_order
        if starts_new:
            current = {"注文番号": len(orders) + 1, "pages": [page_no], **header, **issuer_info}
            orders.append(current)
        else:
            current["pages"].append(page_no)
            for k in ["発注No", "受注No", "日付", "扱い店", "納入先", "納入先郵便番号", "納入先住所", "発注者", "発行会社", "発行元郵便番号", "発行元住所", "発行元TEL", "発行元FAX", "発行部署"]:
                value = header.get(k, "") or issuer_info.get(k, "")
                if not current.get(k) and value:
                    current[k] = value
    page_to_order = {}
    for o in orders:
        for p in o["pages"]:
            page_to_order[p] = o
    logging.info("order_groups=%s", [{k: v for k, v in o.items() if k != "pages"} for o in orders])
    return orders, page_to_order


def get_page_column_profile(page_lines: List[dict]) -> dict:
    profile = {"name_x": None, "qty_x": None, "unit_x": None, "price_x": None, "amount_x": None, "order_x": None, "due_x": None}
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
    return any(k in text for k in ["品名", "仕様", "荷姿数量", "荷姿単位", "単価", "金額", "客先注文番号", "回答納期"])


def is_noise_number(text: str) -> bool:
    t = norm_text(text)
    if re.fullmatch(r"\d{8,}", t):
        return True
    if re.search(r"\d{4}/\d{2}/\d{2}", t):
        return True
    if re.search(r"P\.\d+|053225|000200", t):
        return True
    return False


def is_noise_or_footer(text: str) -> bool:
    return any(k in text for k in ["下請事業者殿", "支払条件", "合計金額", "回答納期欄", "000200-42", "P.001", "P.002", "P.003", "P.004", "▼", "注文書", "発注先", "扱い店", "納入先", "シーアイマテックス", "MATEX", "TEL", "FAX", "ページ", "日付", "受注No"])


def contains_product_name(text: str) -> bool:
    return "加工品" in text


def contains_dimension(text: str) -> bool:
    return DIM_PATTERN.search(norm_text(text)) is not None


def has_special_order_detail_table(lines: List[dict]) -> bool:
    text_all = " ".join(l["text"] for l in lines)
    header_ok = all(k in text_all for k in ["品名", "仕様", "荷姿", "単価", "金額"])
    has_product_word = any(contains_product_name(l["text"]) for l in lines)
    has_dimension = any(contains_dimension(l["text"]) for l in lines)
    logging.info("special_table_detection header_ok=%s has_product_word=%s has_dimension=%s", header_ok, has_product_word, has_dimension)
    return header_ok and has_product_word and has_dimension


def is_product_start_line(line: dict, lines_same_page: List[dict]) -> bool:
    text = line["text"]
    if contains_product_name(text) and contains_dimension(text):
        return True
    if contains_product_name(text) and not contains_dimension(text):
        page_h = line.get("page_h") or 1
        y_limit = line["y"] + page_h * 0.035
        for other in lines_same_page:
            if other["y"] <= line["y"]:
                continue
            if other["y"] > y_limit:
                break
            if contains_dimension(other["text"]):
                return True
    return False


def split_name_spec(text: str) -> Tuple[str, str]:
    text = norm_text(text)
    m = DIM_PATTERN.search(text)
    if not m:
        return text, ""
    name = re.sub(r"\s+0\.$", "", text[:m.start()].strip()).strip()
    spec = re.sub(r"^15\s*[×]", "0.15 ×", m.group(0).strip()).strip()
    return name, spec


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
        if re.fullmatch(r"\(\d+\)", text) or not re.fullmatch(r"\d+", text) or text == "0":
            continue
        if abs(line["cy"] - label_line["cy"]) > page_h * 0.04:
            continue
        if line["cx"] < label_line["cx"] - page_w * 0.03:
            continue
        candidates.append((line_distance(label_line, line), idx, text))
    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0])
    used_ids.add(candidates[0][1])
    return candidates[0][2]


def extract_number_from_same_or_next(label_regex: str, band_lines: List[dict], used_ids: set) -> str:
    same_line_pattern = re.compile(rf"(?:{label_regex})\s*(\(?\d+\)?)$")
    label_only_pattern = re.compile(rf"^(?:{label_regex})$")
    for idx, line in enumerate(band_lines):
        m = same_line_pattern.search(line["text"])
        if m and m.group(1):
            raw = m.group(1).strip()
            if raw.startswith("(") and raw.endswith(")"):
                continue
            clean = raw.strip("()")
            if clean == "0":
                continue
            used_ids.add(idx)
            return clean
    for idx, line in enumerate(band_lines):
        if label_only_pattern.fullmatch(line["text"]):
            val = nearest_numeric_after_label(line, band_lines, used_ids)
            if val and val != "0":
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
    for i, line in enumerate(band_lines):
        if line["text"] == "摘要":
            for nxt in band_lines[i + 1:i + 5]:
                t = nxt["text"]
                if t not in ["希望納期", "摘要"] and not re.fullmatch(r"\d+", t) and not is_noise_or_footer(t):
                    return t
    for line in band_lines:
        if ("試験" in line["text"] or "様" in line["text"]) and not is_noise_or_footer(line["text"]):
            return line["text"]
    return ""


def extract_quantity_and_unit(band_lines: List[dict], used_ids: set, profile: dict) -> Tuple[str, str]:
    qty_x, unit_x = profile.get("qty_x"), profile.get("unit_x")
    unit_candidates = []
    for idx, line in enumerate(band_lines):
        t = line["text"]
        if t in UNIT_WORDS or (any(u in t for u in UNIT_WORDS) and not any(k in t for k in ["加工", "原", "反"])):
            unit_candidates.append((abs(line["cx"] - unit_x), idx, line))
    if not unit_candidates:
        return "", ""
    unit_candidates.sort(key=lambda x: x[0])
    _, unit_idx, unit_line = unit_candidates[0]
    unit = next((u for u in UNIT_WORDS if u in unit_line["text"]), "")
    candidates = []
    page_h = unit_line.get("page_h") or 1
    for idx, line in enumerate(band_lines):
        if idx in used_ids:
            continue
        t = line["text"]
        if is_noise_number(t) or not re.fullmatch(r"\d+", t):
            continue
        n = int(t)
        if not (1 <= n <= 99):
            continue
        if qty_x is not None and abs(line["cx"] - qty_x) > (line.get("page_w") or 1) * 0.08:
            continue
        if abs(line["cy"] - unit_line["cy"]) > page_h * 0.04:
            continue
        candidates.append((abs(line["cx"] - qty_x) if qty_x is not None else line_distance(unit_line, line), idx, t))
    qty = ""
    if candidates:
        candidates.sort(key=lambda x: x[0])
        _, qidx, qty = candidates[0]
        used_ids.add(qidx)
    used_ids.add(unit_idx)
    return qty, unit


def extract_price_amount(band_lines: List[dict], used_ids: set, profile: dict) -> Tuple[str, str]:
    price_x, amount_x, order_x = profile.get("price_x"), profile.get("amount_x"), profile.get("order_x")
    candidates = []
    for idx, line in enumerate(band_lines):
        if idx in used_ids:
            continue
        t = line["text"]
        if is_noise_number(t) or any(k in t for k in ["原", "反", "加工", "賃", "貸", "貨", "愛", "價"]):
            continue
        nums = re.findall(r"\d+", t)
        if len(nums) >= 2:
            price, amount = nums[0], nums[1]
            if len(price) < 4 or amount != "0":
                continue
            page_w = line.get("page_w") or 1
            if price_x is not None and abs(line["cx"] - price_x) > page_w * 0.18:
                continue
            if order_x is not None and line["cx"] > order_x:
                continue
            candidates.append((abs(line["cx"] - price_x), line["y"], idx, price, amount))
    if candidates:
        candidates.sort(key=lambda x: (x[0], x[1]))
        _, _, idx, price, amount = candidates[0]
        used_ids.add(idx)
        return price, amount
    standalone = []
    for idx, line in enumerate(band_lines):
        if idx in used_ids:
            continue
        t = line["text"]
        if is_noise_number(t) or any(k in t for k in ["原", "反", "加工", "賃", "貸", "貨", "愛", "價"]):
            continue
        if not re.fullmatch(r"\d+", t) or len(t) < 4:
            continue
        page_w = line.get("page_w") or 1
        if price_x is not None and abs(line["cx"] - price_x) > page_w * 0.12:
            continue
        if order_x is not None and line["cx"] > order_x:
            continue
        standalone.append((abs(line["cx"] - price_x), line["y"], idx, t))
    if not standalone:
        return "", ""
    standalone.sort(key=lambda x: (x[0], x[1]))
    _, _, price_idx, price = standalone[0]
    used_ids.add(price_idx)
    amount = ""
    price_line = band_lines[price_idx]
    page_h = price_line.get("page_h") or 1
    zeros = []
    for j, zline in enumerate(band_lines):
        if j in used_ids or zline["text"] != "0":
            continue
        if abs(zline["cy"] - price_line["cy"]) > page_h * 0.04:
            continue
        zeros.append((abs(zline["cx"] - amount_x) if amount_x is not None else line_distance(price_line, zline), j))
    if zeros:
        zeros.sort(key=lambda x: x[0])
        _, zero_idx = zeros[0]
        amount = "0"
        used_ids.add(zero_idx)
    return price, amount


def resolve_product_name_from_band(product_line: dict, band_lines: List[dict]) -> str:
    text = product_line["text"].strip()
    if text != "加工品":
        return text
    candidates = []
    for line in band_lines:
        t = line["text"].strip()
        if line["y"] >= product_line["y"]:
            continue
        if not t or contains_dimension(t) or is_header_line(t) or is_noise_or_footer(t) or re.fullmatch(r"\d+", t):
            continue
        if any(k in t for k in ["原", "反", "加工賃", "摘要", "希望納期"]):
            continue
        candidates.append((abs(product_line["y"] - line["y"]), t))
    if candidates:
        candidates.sort(key=lambda x: x[0])
        return f"{candidates[0][1]} 加工品"
    return text


def build_product_from_band(product_line: dict, band_lines: List[dict], profile: dict) -> Dict[str, str]:
    used_ids: set = set()
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
                m = DIM_PATTERN.search(line["text"])
                if m:
                    spec = m.group(0).strip()
                    used_ids.add(idx)
                    break
    return {
        "品名": name,
        "仕様": spec,
        "荷姿数量": extract_quantity_and_unit(band_lines, used_ids, profile)[0],
        "荷姿単位": extract_quantity_and_unit(band_lines, used_ids, profile)[1],
        "単価": "",
        "金額": "",
        "客先注文番号": extract_order_number(band_lines),
        "回答納期": "",
        "希望納期": "",
        "摘要": extract_remark(band_lines),
        "原反": "",
        "加工賃": "",
    } | _price_gen_proc(band_lines, used_ids, profile)


def _price_gen_proc(band_lines: List[dict], used_ids: set, profile: dict) -> Dict[str, str]:
    price, amount = extract_price_amount(band_lines, used_ids, profile)
    gen = extract_number_from_same_or_next(r"原\s*反|原区|原\s*区", band_lines, used_ids)
    proc = extract_number_from_same_or_next(r"加工[賃貸貨愛價]?", band_lines, used_ids)
    return {"単価": price, "金額": amount, "原反": gen, "加工賃": proc}


def extract_order_rows_from_read_lines(analyze_result: dict, file_name: str) -> List[Dict[str, Any]]:
    lines = get_ocr_lines(analyze_result)
    if not lines or not has_special_order_detail_table(lines):
        return []
    orders, page_to_order = build_order_groups(lines)
    item_counts = {o["注文番号"]: 0 for o in orders}
    rows = []
    for page_no in sorted(set(l["page"] for l in lines)):
        page_lines = [l for l in lines if l["page"] == page_no]
        if not page_lines or page_no not in page_to_order:
            continue
        order = page_to_order[page_no]
        profile = get_page_column_profile(page_lines)
        printed_page_label = extract_document_page_label(page_lines)
        order_pages = sorted(order.get("pages", []))
        computed_page_label = f"{order_pages.index(page_no) + 1}/{len(order_pages)}" if page_no in order_pages else str(page_no)
        page_label = printed_page_label or computed_page_label
        logging.info("page=%s document_page=%s column_profile=%s", page_no, page_label, profile)

        product_starts: List[dict] = []
        for line in page_lines:
            if is_header_line(line["text"]) or is_noise_or_footer(line["text"]):
                continue
            if is_product_start_line(line, page_lines):
                product_starts.append(line)
        deduped = []
        for line in product_starts:
            if not deduped or abs(line["y"] - deduped[-1]["y"]) > (line.get("page_h") or 1) * 0.025:
                deduped.append(line)
        product_starts = deduped
        logging.info("page=%s product_start_count=%s order=%s", page_no, len(product_starts), order["注文番号"])

        for idx, start_line in enumerate(product_starts):
            next_start = product_starts[idx + 1] if idx + 1 < len(product_starts) else None
            page_h = start_line.get("page_h") or 1
            band_start_y = start_line["y"] - page_h * 0.004
            if next_start:
                band_end_y = next_start["y"] - page_h * BAND_END_MARGIN_RATIO
            else:
                footer_candidates = [l for l in page_lines if l["y"] > start_line["y"] and any(k in l["text"] for k in ["合計金額", "下請事業者殿", "回答納期欄", "▼", "P."])]
                band_end_y = (min(l["y"] for l in footer_candidates) - page_h * 0.004) if footer_candidates else start_line["y"] + page_h * 0.08
            band_lines = [l for l in page_lines if band_start_y <= l["y"] < band_end_y and not is_header_line(l["text"]) and not is_noise_or_footer(l["text"])]
            band_lines.sort(key=lambda v: (v["y"], v["x"]))
            if not any(contains_product_name(l["text"]) for l in band_lines):
                continue
            item_counts[order["注文番号"]] += 1
            detail = build_product_from_band(start_line, band_lines, profile)
            row = {
                "注文番号": order["注文番号"],
                "注文内明細番号": item_counts[order["注文番号"]],
                "発注No": order.get("発注No", ""),
                "受注No": order.get("受注No", ""),
                "ページ": page_label,
                "扱い店": order.get("扱い店", ""),
                "納入先": order.get("納入先", ""),
                "納入先郵便番号": order.get("納入先郵便番号", ""),
                "納入先住所": order.get("納入先住所", ""),
                "発注者": order.get("発注者", ""),
                "日付": order.get("日付", ""),
                "発行会社": order.get("発行会社", ""),
                "発行元郵便番号": order.get("発行元郵便番号", ""),
                "発行元住所": order.get("発行元住所", ""),
                "発行元TEL": order.get("発行元TEL", ""),
                "発行元FAX": order.get("発行元FAX", ""),
                "発行部署": order.get("発行部署", ""),
                **detail,
                "ファイル参考": file_name,
            }
            rows.append(row)
    logging.info("horizontal_rows_count=%s", len(rows))
    return rows


def split_key_order_suffix(key: str) -> Tuple[str, Optional[int]]:
    m = re.match(r"^(.*)_(\d+)$", str(key))
    return (m.group(1), int(m.group(2))) if m and m.group(1) else (str(key), None)


def enrich_order_rows_with_gpt_columns(rows: List[Dict[str, Any]], ai_result: dict) -> Tuple[List[Dict[str, Any]], List[str]]:
    flat = flatten_json(ai_result)
    product_key_pattern = re.compile(r"^(品名|仕様|荷姿数量|荷姿単位|単価|単価\(単位\)|金額|客先注文番号|回答納期|希望納期|摘要|原|原反|原区|加工賃|加工貸|加工貨|加工愛|加工價)_\d+$")
    product_base_keys = {"品名", "仕様", "荷姿数量", "荷姿単位", "単価", "単価(単位)", "金額", "客先注文番号", "回答納期", "希望納期", "摘要", "原", "原反", "原区", "加工賃", "加工貸", "加工貨", "加工愛", "加工價"}
    fixed_order_keys = {"発注No", "受注No", "ページ", "扱い店", "納入先", "納入先郵便番号", "納入先住所", "発注者", "日付", "発行会社", "発行元郵便番号", "発行元住所", "発行元TEL", "発行元FAX", "発行部署"}
    global_values: Dict[str, Any] = {}
    order_values: Dict[int, Dict[str, Any]] = {}
    extra_headers: List[str] = []
    for raw_key, value in flat.items():
        key = str(raw_key)
        if product_key_pattern.match(key):
            continue
        base_key, order_no = split_key_order_suffix(key)
        if base_key in product_base_keys or base_key == "ファイル参考":
            continue
        if order_no is None:
            global_values[base_key] = value
        else:
            order_values.setdefault(order_no, {})[base_key] = value
        if should_skip_gpt_extra_header(base_key):
            continue

        if base_key not in HORIZONTAL_HEADERS and base_key not in fixed_order_keys and base_key not in extra_headers:
            extra_headers.append(base_key)
    enriched = []
    for row in rows:
        merged = dict(row)
        try:
            order_no = int(row.get("注文番号") or 1)
        except Exception:
            order_no = 1
        values = order_values.get(order_no, {})
        for fixed_key in fixed_order_keys:
            current = str(merged.get(fixed_key, "") or "")
            if fixed_key == "納入先":
                value = values.get(fixed_key)
                if value not in (None, "") and (current == "" or current.startswith("〒")) and not str(value).startswith("〒"):
                    merged[fixed_key] = value
                continue
            value = values.get(fixed_key)
            if value in (None, ""):
                value = global_values.get(fixed_key, "")
            if value in (None, ""):
                continue
            if fixed_key == "ページ":
                if "/" not in current and "/" in str(value):
                    merged[fixed_key] = value
                continue
            if current == "":
                merged[fixed_key] = value
        for extra_key in extra_headers:
            value = values.get(extra_key)
            if value in (None, ""):
                value = global_values.get(extra_key, "")
            merged[extra_key] = value if value is not None else ""
        enriched.append(merged)
    return enriched, extra_headers


def split_delivery_destination(value: Any) -> Tuple[str, str, str]:
    text = str(value or "").strip()
    if not text:
        return "", "", ""
    m = re.search(r"(〒\s*\d{3}-\d{4})", text)
    if not m:
        return text, "", ""
    name = text[:m.start()].strip()
    postal = m.group(1).replace(" ", "")
    address = text[m.end():].strip()
    if postal == "〒105-0014":
        return name, "", address
    return name, postal, address


def normalize_delivery_columns(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    normalized = []
    for row in rows:
        new_row = dict(row)
        name, postal, address = split_delivery_destination(new_row.get("納入先", ""))
        if name:
            new_row["納入先"] = name
        if postal and not new_row.get("納入先郵便番号"):
            new_row["納入先郵便番号"] = postal
        if address and not new_row.get("納入先住所"):
            new_row["納入先住所"] = address
        if str(new_row.get("納入先", "")).startswith("〒"):
            new_row["納入先"] = ""
        if str(new_row.get("納入先郵便番号", "")).startswith("〒105-0014"):
            new_row["納入先郵便番号"] = ""
        normalized.append(new_row)
    return normalized


def build_generic_dynamic_horizontal(ai_result: dict, file_name: str) -> Tuple[List[Dict[str, Any]], List[str]]:
    flat = flatten_json(ai_result)
    flat["ファイル参考"] = file_name
    headers = [k for k in flat.keys() if k != "ファイル参考"] + ["ファイル参考"]
    return [flat], headers


# -----------------------------
# Azure OpenAI
# -----------------------------

def run_aoai_extraction(ocr_text: str, prompt: str = "") -> Dict[str, Any]:
    url = f"{Config.AOAI_ENDPOINT}/openai/deployments/{Config.AOAI_DEPLOYMENT}/chat/completions?api-version={Config.AOAI_API_VERSION}"
    headers = {"api-key": Config.AOAI_KEY, "Content-Type": "application/json"}
    body = {"messages": [{"role": "system", "content": "Return JSON only"}, {"role": "user", "content": build_user_prompt(ocr_text, prompt)}], "temperature": 0}
    logging.info("Calling Azure OpenAI for structured extraction")
    response = requests.post(url, headers=headers, json=body, timeout=(Config.HTTP_CONNECT_TIMEOUT_SEC, Config.HTTP_READ_TIMEOUT_SEC))
    if response.status_code >= 400:
        raise APIError("Azure OpenAI request failed.", 502, {"status_code": response.status_code, "response": safe_json_or_text(response)})
    content = extract_aoai_content(response.json())
    parsed = parse_json_safely(content)
    if not isinstance(parsed, dict):
        raise APIError("AI output must be a JSON object.", 502, {"type": type(parsed).__name__})
    return parsed


def build_user_prompt(ocr_text: str, user_prompt: str) -> str:
    base = """
Extract structured data from the OCR text.
Rules:
- Keep values exactly as written in OCR.
- Return only valid JSON object.
- Do not merge multiple labels into one field.
- If unclear, return empty string.
- Do not infer values that do not have labels in OCR.
- Keep all OCR-extracted useful data.
"""
    return base + "\nUser instructions:\n" + user_prompt + "\n\nOCR:\n" + ocr_text


# -----------------------------
# Output
# -----------------------------

def build_response(rows: List[Dict[str, Any]], headers: List[str], output_format: str, output_filename: str, sheet_name: str) -> Response:
    if output_format == "json":
        return Response(content=json.dumps({"headers": headers, "rows": rows}, ensure_ascii=False, indent=2), media_type="application/json; charset=utf-8")
    if output_format == "csv":
        csv_text = build_rows_csv(rows, headers)
        return Response(content="\ufeff" + csv_text, media_type="text/csv; charset=utf-8", headers={"Content-Disposition": f"attachment; filename={output_filename}"})
    if output_format == "txt":
        txt = json.dumps({"headers": headers, "rows": rows}, ensure_ascii=False, indent=2)
        return Response(content=txt, media_type="text/plain; charset=utf-8", headers={"Content-Disposition": f"attachment; filename={output_filename}"})
    xlsx_bytes = build_rows_xlsx(rows, headers, sheet_name)
    return StreamingResponse(io.BytesIO(xlsx_bytes), media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", headers={"Content-Disposition": f"attachment; filename={output_filename}"})


def build_rows_csv(rows: List[Dict[str, Any]], headers: List[str]) -> str:
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=headers, extrasaction="ignore", lineterminator="\r\n")
    writer.writeheader()
    for row in rows:
        out_row = {}
        for h in headers:
            value = row.get(h, "")
            # Prevent Excel from turning 1/2, 2/2, 1/1 into dates when opening CSV.
            if h == "ページ" and isinstance(value, str) and "/" in value:
                value = f'="{value}"'
            out_row[h] = value
        writer.writerow(out_row)
    return output.getvalue()


def build_rows_xlsx(rows: List[Dict[str, Any]], headers: List[str], sheet_name: str = "結果") -> bytes:
    wb = Workbook()
    ws = wb.active
    ws.title = sheet_name
    ws.append(headers)
    for row in rows:
        ws.append([row.get(h, "") for h in headers])
    for row_cells in ws.iter_rows():
        for cell in row_cells:
            cell.number_format = "@"
    header_fill = PatternFill("solid", fgColor="1F4E78")
    header_font = Font(color="FFFFFF", bold=True)
    for cell in ws[1]:
        cell.fill = header_fill
        cell.font = header_font
        cell.alignment = Alignment(horizontal="center")
    ws.freeze_panes = "A2"
    ws.auto_filter.ref = ws.dimensions
    for col_idx in range(1, len(headers) + 1):
        max_len = 0
        for row_idx in range(1, ws.max_row + 1):
            val = ws.cell(row=row_idx, column=col_idx).value
            max_len = max(max_len, len(str(val)) if val is not None else 0)
        ws.column_dimensions[get_column_letter(col_idx)].width = min(max_len + 2, 32)
    output = io.BytesIO()
    wb.save(output)
    output.seek(0)
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
            items.update(flatten_json(v, new_key, sep))
        elif isinstance(v, list):
            items[new_key] = json.dumps(v, ensure_ascii=False)
        else:
            items[new_key] = v
    return items


def clean_key(key: str) -> str:
    return key.replace('"', "").replace("'", "").replace(" ", "").replace("\n", "").replace("\t", "").strip()


def parse_json_safely(content: str) -> Any:
    for candidate in [content, strip_markdown_fences(content)]:
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass
    cleaned = strip_markdown_fences(content)
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


if __name__ == "__main__":
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("app:app", host="127.0.0.1", port=port, reload=True, http="httptools")
