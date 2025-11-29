# tools/image_ocr.py
import os
import base64
import time
import logging
from typing import Optional, Dict, Any
from urllib.parse import urljoin

import requests
from google import genai

logger = logging.getLogger(__name__)

# reuse existing client
client = genai.Client(api_key=os.environ.get("GOOGLE_API_KEY", ""))

def _detect_mime_type_from_headers(headers: Dict[str, Any]) -> str:
    ct = headers.get("content-type", "")
    if ";" in ct:
        ct = ct.split(";", 1)[0].strip()
    # fallback defaults
    if ct in ("image/png", "image/jpeg", "image/jpg", "image/webp", "image/gif"):
        return ct
    # if unknown, default to png (safe choice for base64 inline)
    return "image/png"

def image_ocr(
    image_url: str,
    base_url: Optional[str] = None,
    max_retries: int = 3,
    backoff_seconds: float = 1.0,
    request_timeout: int = 20,
) -> Dict[str, Any]:
    """
    Robust image OCR using Gemini Vision.

    Returns a dict:
      {
        "success": bool,
        "text": str | None,        # extracted text on success
        "error": str | None,       # error code or message on failure
        "url": str                 # final resolved URL tried
      }

    Notes:
    - If image_url is relative (e.g. "sample_image.png"), set base_url to the page URL
      so the function resolves the absolute path.
    - This function never raises network/requests exceptions; instead it returns
      structured failure info for the caller to handle.
    """
    if base_url:
        image_url = urljoin(base_url, image_url)

    # Simple retry loop for transient network issues
    last_exc = None
    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.get(image_url, timeout=request_timeout)
            # do not raise_for_status(); we handle status codes explicitly
            break
        except requests.RequestException as e:
            last_exc = e
            logger.warning("Attempt %d: failed to fetch image %s: %s", attempt, image_url, e)
            if attempt < max_retries:
                time.sleep(backoff_seconds * attempt)
            else:
                return {
                    "success": False,
                    "text": None,
                    "error": f"request_exception: {str(e)}",
                    "url": image_url,
                }

    # At this point we have a resp object (or returned due to exception)
    if resp.status_code != 200:
        # 404 or other http errors => return a structured failure (no exceptions)
        logger.info("Image fetch returned status %s for %s", resp.status_code, image_url)
        return {
            "success": False,
            "text": None,
            "error": f"http_{resp.status_code}",
            "url": image_url,
        }

    image_bytes = resp.content
    if not image_bytes:
        return {
            "success": False,
            "text": None,
            "error": "empty_content",
            "url": image_url,
        }

    # Determine a sensible mime type from headers (fallback to png)
    mime_type = _detect_mime_type_from_headers(resp.headers)

    inline_data = {
        "mime_type": mime_type,
        "data": base64.b64encode(image_bytes).decode("ascii"),
    }

    prompt = (
        "Read the text in this image and return ONLY the exact secret code, "
        "without extra words, quotes, or punctuation."
    )

    # Call Gemini Vision / generate_content safely
    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[
                {
                    "role": "user",
                    "parts": [
                        {"text": prompt},
                        {"inline_data": inline_data},
                    ],
                }
            ],
        )
        # `response.text` is used in your original code — keep same behavior if present
        text = (getattr(response, "text", None) or "") or ""
        text = text.strip()
        if text == "":
            # empty but successful response
            return {
                "success": True,
                "text": "",
                "error": "empty_response",
                "url": image_url,
            }
        return {"success": True, "text": text, "error": None, "url": image_url}
    except Exception as e:
        logger.exception("Gemini OCR call failed for %s", image_url)
        return {"success": False, "text": None, "error": f"gemini_error: {e}", "url": image_url}


# Compatibility wrapper that keeps your original signature (returns a string)
def image_ocr_text(image_url: str, base_url: Optional[str] = None) -> str:
    """
    Backwards-compatible wrapper. Returns extracted text or empty string on failure.
    """
    result = image_ocr(image_url=image_url, base_url=base_url)
    if result.get("success"):
        return result.get("text", "") or ""
    # optionally log full result for debugging
    logger.debug("image_ocr_text returning empty on failure: %s", result)
    return ""
