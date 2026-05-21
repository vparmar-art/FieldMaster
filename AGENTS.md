# FieldMaster Agent Guide

## App Purpose

This is a small Flask web app for plant disease detection from a single uploaded image. It sends the uploaded image to Google Gemini, expects Gemini to return JSON detections, draws red bounding boxes with Pillow, and renders the original image, annotated image, and JSON result on one page.

## Project Layout

- `app.py`: Flask routes, upload validation, Gemini API call, JSON parsing/sanitizing, and Pillow annotation.
- `templates/index.html`: Single-page upload form and results UI.
- `requirements.txt`: Python dependencies.
- `.env.example`: Template for required environment variables.
- `.env`: Local environment file. It is ignored by Git and should not contain committed secrets.
- `uploads/`: Runtime upload storage, ignored by Git.
- `uploads/annotated/`: Runtime annotated image storage, ignored by Git.

## Runtime Configuration

The app loads environment variables from `.env` using `python-dotenv`.

Required variable:

```bash
GEMINI_API_KEY="your_key_here"
```

Do not hardcode API keys in source files. Do not print real API keys in logs, responses, or documentation.

## Setup And Run

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Edit `.env`, then run:

```bash
flask run
```

Default local URL:

```bash
http://127.0.0.1:5000
```

## Gemini Behavior

- SDK: `google-genai`
- Model: `gemini-2.5-flash`
- API key source: `GEMINI_API_KEY`
- The uploaded image is sent as bytes with the detected MIME type.
- Temporary Gemini 429/503-style overload errors are retried three times with short backoff.
- The prompt requires Gemini to return only valid JSON:

```json
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
```

If no disease is visible, the expected response is:

```json
{"detections":[]}
```

## Important Implementation Notes

- Accepts one uploaded image per request.
- Allowed extensions: `png`, `jpg`, `jpeg`, `webp`.
- Max upload size: 16 MB.
- Uploads are verified with Pillow before calling Gemini.
- Gemini output is treated as untrusted text. `extract_json_object()` parses JSON defensively, and `normalize_detections()` validates/sanitizes detections before rendering or drawing.
- Bounding box coordinates are clamped to the original image dimensions.
- Annotated images are always saved as JPEG under `uploads/annotated/`.

## Verification

Useful checks:

```bash
source venv/bin/activate
python -m compileall app.py
flask run
```

Without a real `GEMINI_API_KEY`, the upload flow should render a page error rather than crash.
