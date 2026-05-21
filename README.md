# Plant Disease Detection Flask App

A simple Flask app that accepts one uploaded plant image, sends it to Google Gemini, draws red bounding boxes for detected disease or unhealthy patches, and shows the original image, annotated image, and JSON result.

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Edit `.env` and set your Gemini API key:

```bash
GEMINI_API_KEY="your_key_here"
```

Run the app:

```bash
flask run
```

Open the local Flask URL shown in your terminal, usually `http://127.0.0.1:5000`.

## Notes

- Uploads are stored locally in `uploads/`.
- Annotated images are stored in `uploads/annotated/`.
- The app loads `.env` with `python-dotenv` and reads `GEMINI_API_KEY`.
- The Gemini request uses `gemini-2.5-flash`.
- Temporary Gemini 429/503 overload errors are retried before showing a friendly error.
