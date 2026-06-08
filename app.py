import json
import logging
import os
import time
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import requests
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.responses import JSONResponse
from requests import Response
from starlette.exceptions import HTTPException as StarletteHTTPException


load_dotenv()
APP_VERSION = "2026-06-05.2"


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
    return JSONResponse(
        status_code=504,
        content={"error": "External service timeout.", "details": {"reason": str(err)}},
    )


@app.exception_handler(requests.RequestException)
async def request_exception_handler(_: Request, err: requests.RequestException) -> JSONResponse:
    logging.exception("Request exception while calling external service")
    return JSONResponse(
        status_code=502,
        content={"error": "External service request failed.", "details": {"reason": str(err)}},
    )


@app.exception_handler(StarletteHTTPException)
async def starlette_http_exception_handler(_: Request, err: StarletteHTTPException) -> JSONResponse:
    return JSONResponse(
        status_code=err.status_code,
        content={"error": "HTTP error", "details": {"reason": str(err.detail)}},
    )


@app.exception_handler(Exception)
async def unexpected_error_handler(_: Request, err: Exception) -> JSONResponse:
    logging.exception("Unexpected server error")
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error.", "details": {"reason": str(err)}},
    )


@app.get("/health")
async def health() -> Dict[str, str]:
    return {
        "status": "ok",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "version": APP_VERSION,
    }


@app.post("/process")
async def process_document(
    file: UploadFile = File(...),
    prompt: Optional[str] = Form(default=""),
) -> Dict[str, Any]:
    validate_required_env()

    if not file or not file.filename:
        raise APIError("No file provided.", 400)

    filename = os.path.basename(file.filename)
    content_type = file.content_type or "application/octet-stream"

    if not is_allowed_file(filename, content_type):
        raise APIError("Unsupported file type. Allowed: PDF or image formats.", 400)

    file_bytes = await file.read()
    if not file_bytes:
        raise APIError("Uploaded file is empty.", 400)

    clean_prompt = (prompt or "").strip()

    logging.info("Processing request for file=%s, content_type=%s", filename, content_type)

    ocr_text = run_document_intelligence_ocr(file_bytes=file_bytes, content_type=content_type)
    ai_result = run_aoai_extraction(ocr_text=ocr_text, prompt=clean_prompt)

    data = {
        "ocr_text": ocr_text,
        "ai_result": ai_result
    }

    return Response(
        content=json.dumps(data, ensure_ascii=False, indent=2),
        media_type="application/json"
    )



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


def run_document_intelligence_ocr(file_bytes: bytes, content_type: str) -> str:
    url = (
        f"{Config.DOCINT_ENDPOINT}/documentintelligence/documentModels/"
        f"{Config.DOCINT_MODEL_ID}:analyze?api-version={Config.DOCINT_API_VERSION}"
    )
    headers = {
        "Ocp-Apim-Subscription-Key": Config.DOCINT_KEY,
        "Content-Type": content_type,
    }

    logging.info("Submitting file to Azure Document Intelligence OCR")
    response = requests.post(
        url,
        headers=headers,
        data=file_bytes,
        timeout=(Config.HTTP_CONNECT_TIMEOUT_SEC, Config.HTTP_READ_TIMEOUT_SEC),
    )

    if response.status_code not in (200, 202):
        raise APIError(
            "Document Intelligence analyze request failed.",
            502,
            {
                "status_code": response.status_code,
                "response": safe_json_or_text(response),
            },
        )

    operation_location = response.headers.get("Operation-Location")
    if not operation_location:
        payload = response.json() if response.content else {}
        status = str(payload.get("status", "")).lower()
        if status == "succeeded":
            text = (((payload.get("analyzeResult") or {}).get("content")) or "").strip()
            if not text:
                raise APIError("OCR completed but analyzeResult.content was empty.", 502)
            return text

        raise APIError(
            "Missing Operation-Location header from Document Intelligence response.",
            502,
            {"response": safe_json_or_text(response)},
        )

    return poll_docint_result(operation_location)


def poll_docint_result(operation_location: str) -> str:
    headers = {"Ocp-Apim-Subscription-Key": Config.DOCINT_KEY}

    interval = Config.DOCINT_POLL_INTERVAL_SEC
    for attempt in range(1, Config.DOCINT_POLL_MAX_RETRIES + 1):
        logging.info("Polling OCR operation (attempt %s/%s)", attempt, Config.DOCINT_POLL_MAX_RETRIES)

        response = requests.get(
            operation_location,
            headers=headers,
            timeout=(Config.HTTP_CONNECT_TIMEOUT_SEC, Config.HTTP_READ_TIMEOUT_SEC),
        )

        if response.status_code >= 400:
            raise APIError(
                "Failed while polling Document Intelligence operation.",
                502,
                {
                    "status_code": response.status_code,
                    "response": safe_json_or_text(response),
                    "operation_location": operation_location,
                },
            )

        payload = response.json()
        status = str(payload.get("status", "")).lower()

        if status == "succeeded":
            text = (((payload.get("analyzeResult") or {}).get("content")) or "").strip()
            if not text:
                raise APIError("OCR succeeded but analyzeResult.content was empty.", 502)
            return text

        if status in {"failed", "canceled", "cancelled"}:
            raise APIError(
                "Document Intelligence operation did not succeed.",
                502,
                {"status": status, "response": payload},
            )

        if attempt < Config.DOCINT_POLL_MAX_RETRIES:
            time.sleep(interval)
            interval = min(
                interval * Config.DOCINT_POLL_BACKOFF_MULTIPLIER,
                Config.DOCINT_POLL_MAX_INTERVAL_SEC,
            )

    raise APIError(
        "Document Intelligence polling timed out before completion.",
        504,
        {"max_retries": Config.DOCINT_POLL_MAX_RETRIES},
    )


def run_aoai_extraction(ocr_text: str, prompt: str = "") -> Dict[str, Any]:
    url = (
        f"{Config.AOAI_ENDPOINT}/openai/deployments/{Config.AOAI_DEPLOYMENT}/"
        f"chat/completions?api-version={Config.AOAI_API_VERSION}"
    )
    headers = {
        "api-key": Config.AOAI_KEY,
        "Content-Type": "application/json",
    }

    user_instruction = build_user_prompt(ocr_text=ocr_text, prompt=prompt)
    body = {
        "messages": [
            {"role": "system", "content": "Return JSON only"},
            {"role": "user", "content": user_instruction},
        ],
        "temperature": 0,
    }

    logging.info("Calling Azure OpenAI for structured extraction")
    response = requests.post(
        url,
        headers=headers,
        json=body,
        timeout=(Config.HTTP_CONNECT_TIMEOUT_SEC, Config.HTTP_READ_TIMEOUT_SEC),
    )

    if response.status_code >= 400:
        raise APIError(
            "Azure OpenAI request failed.",
            502,
            {
                "status_code": response.status_code,
                "response": safe_json_or_text(response),
            },
        )

    payload = response.json()
    content = extract_aoai_content(payload)
    parsed = parse_json_safely(content)

    if not isinstance(parsed, dict):
        raise APIError(
            "AI output must be a JSON object.",
            502,
            {"type": type(parsed).__name__},
        )

    return parsed


def build_user_prompt(ocr_text: str, prompt: str) -> str:
    base = "Extract structured information from the OCR text below and return valid JSON."
    if prompt:
        base = f"{base}\n\nAdditional extraction instructions:\n{prompt.strip()}"

    return f"{base}\n\nOCR Text:\n{ocr_text}"


def extract_aoai_content(payload: Dict[str, Any]) -> str:
    try:
        return payload["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise APIError(
            "Unexpected Azure OpenAI response shape.",
            502,
            {"response": payload},
        ) from exc


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


def safe_json_or_text(response: Response) -> Any:
    try:
        return response.json()
    except ValueError:
        return response.text[:4000]


if __name__ == "__main__":
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)
