# =========================================================
# CORE IMPORTS (SAFE)
# =========================================================

import os
import numpy as np

# Lazy imports (avoid uvicorn crash if missing libs)
fitz = None
faiss = None
pytesseract = None
Document = None
VideoFileClip = None
SentenceTransformer = None

from supabase import create_client
from openai import OpenAI
import assemblyai as aai
import google.generativeai as genai


# =========================================================
# CONFIG
# =========================================================

SUPABASE_URL = "https://tbpdhybqbjucoxdizlgw.supabase.co"
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "YOUR_SUPABASE_KEY")

ASSEMBLYAI_API_KEY = os.getenv("ASSEMBLYAI_API_KEY", "YOUR_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "YOUR_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "YOUR_KEY")


# =========================================================
# INIT (SAFE)
# =========================================================

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
client = OpenAI(api_key=OPENAI_API_KEY)

aai.settings.api_key = ASSEMBLYAI_API_KEY

genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel("gemini-1.5-flash")

# FAISS safe init
index = None
embed_model = None


def init_models():
    global faiss, SentenceTransformer, index, embed_model

    import faiss
    from sentence_transformers import SentenceTransformer

    embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    index = faiss.IndexFlatL2(384)


# =========================================================
# SCENES
# =========================================================

SCENES = {
    "warm_to_deep": "FIRST MEMORY emotional depth analysis",
    "scene_first": "ORDINARY DAY narrative reconstruction",
    "relational_lens": "RELATIONSHIPS emotional mapping"
}


# =========================================================
# FILE PROCESSORS (SAFE IMPORTS)
# =========================================================

def process_pdf(path):
    import fitz
    pdf = fitz.open(path)
    return "\n".join(page.get_text() for page in pdf)


def process_docx(path):
    from docx import Document
    doc = Document(path)
    return "\n".join(p.text for p in doc.paragraphs)


def process_txt(path):
    return open(path, encoding="utf-8").read()


def process_image(path):
    import pytesseract
    from PIL import Image
    return pytesseract.image_to_string(Image.open(path))


def process_video(path):
    from moviepy.editor import VideoFileClip

    temp_audio = "temp.wav"
    clip = VideoFileClip(path)

    if clip.audio is None:
        return None

    clip.audio.write_audiofile(temp_audio)
    return transcribe_audio(temp_audio)


# =========================================================
# TRANSCRIPTION
# =========================================================

def transcribe_audio(path):
    transcriber = aai.Transcriber()

    config = aai.TranscriptionConfig(
        speech_models=["universal-3-pro"],
        punctuate=True,
        format_text=True
    )

    result = transcriber.transcribe(path, config=config)
    return result.text if result.text else None


# =========================================================
# GPT-4o
# =========================================================

def analyze_gpt4o(text, scene):

    prompt = f"""
Scene:
{SCENES.get(scene,"")}

Analyze:
- emotions
- relationships
- personality
- narrative

CONTENT:
{text[:8000]}
"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a memory AI system."},
                {"role": "user", "content": prompt}
            ]
        )

        return response.choices[0].message.content

    except Exception as e:
        return f"GPT-4o error: {e}"


# =========================================================
# PIPELINE CORE
# =========================================================

def run(file_path, scene="scene_first"):

    ext = file_path.split(".")[-1].lower()

    if ext in ["mp3", "wav", "m4a"]:
        text = transcribe_audio(file_path)

    elif ext == "pdf":
        text = process_pdf(file_path)

    elif ext == "docx":
        text = process_docx(file_path)

    elif ext == "txt":
        text = process_txt(file_path)

    elif ext in ["jpg", "jpeg", "png"]:
        text = process_image(file_path)

    elif ext in ["mp4", "mov"]:
        text = process_video(file_path)

    else:
        text = None

    if not text:
        return {"error": "No content extracted"}

    result = analyze_gpt4o(text, scene)

    # lazy init embeddings
    try:
        if embed_model is None:
            init_models()

        vec = np.array([embed_model.encode(text)]).astype("float32")
        index.add(vec)

    except:
        pass

    # supabase save
    try:
        supabase.table("documents").insert({
            "type_name": "memory_pipeline",
            "content": text,
            "mini_analysis": result,
            "question_set": scene
        }).execute()
    except:
        pass

    return {"result": result}
