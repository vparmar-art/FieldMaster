import json
import os
import re
import time
import uuid
from pathlib import Path

from dotenv import load_dotenv
from flask import Flask, render_template, request, send_from_directory
from google import genai
from google.genai import types
from PIL import Image, ImageDraw, ImageFont, UnidentifiedImageError
from werkzeug.exceptions import RequestEntityTooLarge


BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
ANNOTATED_DIR = UPLOAD_DIR / "annotated"
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp"}
MAX_CONTENT_LENGTH = 16 * 1024 * 1024
MODEL_NAME = "gemini-2.5-flash"
GEMINI_MAX_RETRIES = 3
GEMINI_RETRY_DELAY_SECONDS = 2

load_dotenv(BASE_DIR / ".env")

PROMPT = """
Analyze this plant image for visible plant diseases or unhealthy patches.

Return ONLY valid JSON with this exact schema:
{
  "detections": [
    {
      "label": "plant disease or unhealthy patch",
      "confidence": 0.0,
      "box": {
        "x1": 0,
        "y1": 0,
        "x2": 0,
        "y2": 0
      },
      "reason": "short explanation"
    }
  ]
}

Rules:
- Coordinates must be pixel coordinates relative to the original image.
- Use tight boxes around the visible diseased or unhealthy patch.
- confidence must be a number between 0.0 and 1.0.
- If no disease is visible, return {"detections":[]}.
- Do not include markdown, comments, or text outside the JSON object.
""".strip()

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = str(UPLOAD_DIR)
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH


class GeminiTemporaryError(RuntimeError):
    pass


def ensure_upload_dirs() -> None:
    UPLOAD_DIR.mkdir(exist_ok=True)
    ANNOTATED_DIR.mkdir(exist_ok=True)


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def save_uploaded_image(file_storage) -> tuple[Path, str, int, int]:
    if not file_storage or not file_storage.filename:
        raise ValueError("Please choose an image to upload.")

    if not allowed_file(file_storage.filename):
        raise ValueError("Only PNG, JPG, JPEG, and WEBP images are supported.")

    suffix = f".{file_storage.filename.rsplit('.', 1)[1].lower()}"
    filename = f"{uuid.uuid4().hex}{suffix}"
    image_path = UPLOAD_DIR / filename
    file_storage.save(image_path)

    try:
        with Image.open(image_path) as image:
            image.verify()
        with Image.open(image_path) as image:
            width, height = image.size
    except (UnidentifiedImageError, OSError) as exc:
        image_path.unlink(missing_ok=True)
        raise ValueError("The uploaded file is not a valid image.") from exc

    return image_path, filename, width, height


def get_image_mime_type(image_path: Path) -> str:
    with Image.open(image_path) as image:
        image_format = image.format

    if image_format == "JPEG":
        return "image/jpeg"
    return Image.MIME.get(image_format, f"image/{image_format.lower()}")


def extract_json_object(raw_text: str) -> dict:
    if not raw_text:
        raise ValueError("Gemini returned an empty response.")

    cleaned = raw_text.strip()
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", cleaned, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        cleaned = fenced.group(1).strip()

    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        decoder = json.JSONDecoder()
        start = cleaned.find("{")
        if start == -1:
            raise ValueError("Gemini did not return a JSON object.")
        try:
            parsed, _ = decoder.raw_decode(cleaned[start:])
        except json.JSONDecodeError as exc:
            raise ValueError("Gemini returned invalid JSON.") from exc

    if not isinstance(parsed, dict):
        raise ValueError("Gemini JSON must be an object.")

    return parsed


def clamp_int(value, lower: int, upper: int) -> int:
    try:
        number = int(round(float(value)))
    except (TypeError, ValueError):
        number = lower
    return max(lower, min(number, upper))


def normalize_detections(data: dict, width: int, height: int) -> dict:
    detections = data.get("detections")
    if detections is None:
        return {"detections": []}
    if not isinstance(detections, list):
        raise ValueError("Gemini JSON field 'detections' must be a list.")

    normalized = []
    for item in detections:
        if not isinstance(item, dict):
            continue

        box = item.get("box")
        if not isinstance(box, dict):
            continue

        x1 = clamp_int(box.get("x1"), 0, width)
        y1 = clamp_int(box.get("y1"), 0, height)
        x2 = clamp_int(box.get("x2"), 0, width)
        y2 = clamp_int(box.get("y2"), 0, height)

        if x2 < x1:
            x1, x2 = x2, x1
        if y2 < y1:
            y1, y2 = y2, y1
        if x2 == x1 or y2 == y1:
            continue

        try:
            confidence = float(item.get("confidence", 0.0))
        except (TypeError, ValueError):
            confidence = 0.0

        normalized.append(
            {
                "label": str(item.get("label") or "unhealthy patch")[:80],
                "confidence": max(0.0, min(confidence, 1.0)),
                "box": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
                "reason": str(item.get("reason") or "")[:240],
            }
        )

    return {"detections": normalized}


def is_retryable_gemini_error(exc: Exception) -> bool:
    message = str(exc).lower()
    return any(
        token in message
        for token in (
            "503",
            "429",
            "unavailable",
            "resource_exhausted",
            "rate limit",
            "quota",
            "high demand",
        )
    )


def generate_gemini_content(client, image_bytes: bytes, mime_type: str):
    return client.models.generate_content(
        model=MODEL_NAME,
        contents=[
            types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
            PROMPT,
        ],
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            temperature=0,
        ),
    )


def call_gemini(image_path: Path, mime_type: str, width: int, height: int) -> dict:
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not set.")

    client = genai.Client(api_key=api_key)
    image_bytes = image_path.read_bytes()

    last_error = None
    for attempt in range(1, GEMINI_MAX_RETRIES + 1):
        try:
            response = generate_gemini_content(client, image_bytes, mime_type)
            break
        except Exception as exc:
            if not is_retryable_gemini_error(exc):
                raise

            last_error = exc
            if attempt == GEMINI_MAX_RETRIES:
                raise GeminiTemporaryError(
                    "Gemini is temporarily overloaded. Please wait a minute and try again."
                ) from last_error

            time.sleep(GEMINI_RETRY_DELAY_SECONDS * attempt)

    parsed = extract_json_object(response.text)
    return normalize_detections(parsed, width, height)


def draw_annotations(image_path: Path, output_filename: str, result: dict) -> str:
    with Image.open(image_path) as image:
        annotated = image.convert("RGB")

    draw = ImageDraw.Draw(annotated)
    font = ImageFont.load_default()

    for detection in result["detections"]:
        box = detection["box"]
        label = f"{detection['label']} {detection['confidence']:.2f}"
        xy = (box["x1"], box["y1"], box["x2"], box["y2"])

        draw.rectangle(xy, outline="red", width=4)
        text_bbox = draw.textbbox((box["x1"], box["y1"]), label, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        label_top = max(0, box["y1"] - text_height - 8)

        draw.rectangle(
            (box["x1"], label_top, box["x1"] + text_width + 8, label_top + text_height + 8),
            fill="red",
        )
        draw.text((box["x1"] + 4, label_top + 4), label, fill="white", font=font)

    annotated_filename = f"annotated_{output_filename.rsplit('.', 1)[0]}.jpg"
    annotated_path = ANNOTATED_DIR / annotated_filename
    annotated.save(annotated_path, format="JPEG", quality=92)
    return f"annotated/{annotated_filename}"


@app.route("/", methods=["GET", "POST"])
def index():
    ensure_upload_dirs()
    error = None
    result = None
    original_image = None
    annotated_image = None

    if request.method == "POST":
        try:
            image_path, filename, width, height = save_uploaded_image(request.files.get("image"))
            mime_type = get_image_mime_type(image_path)
            result = call_gemini(image_path, mime_type, width, height)
            annotated_image = draw_annotations(image_path, filename, result)
            original_image = filename
        except GeminiTemporaryError as exc:
            error = str(exc)
        except ValueError as exc:
            error = str(exc)
        except Exception as exc:
            error = f"Unable to analyze the image: {exc}"

    return render_template(
        "index.html",
        error=error,
        result=result,
        result_json=json.dumps(result, indent=2) if result else None,
        original_image=original_image,
        annotated_image=annotated_image,
    )


@app.errorhandler(RequestEntityTooLarge)
def handle_large_file(_error):
    return (
        render_template(
            "index.html",
            error="The uploaded file is too large. Please choose an image under 16 MB.",
            result=None,
            result_json=None,
            original_image=None,
            annotated_image=None,
        ),
        413,
    )


@app.route("/uploads/<path:filename>")
def uploaded_file(filename: str):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)


if __name__ == "__main__":
    ensure_upload_dirs()
    app.run(debug=True)
