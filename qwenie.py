import os
os.makedirs("./models", exist_ok=True)
os.environ["HF_HOME"] = "./models"
os.environ["HF_HUB_CACHE"] = "./models"
os.environ.setdefault("HF_HOME", "./models")
os.environ.setdefault(
    "PYTORCH_CUDA_ALLOC_CONF",
    "backend:cudaMallocAsync,expandable_segments:True,max_split_size_mb:128,garbage_collection_threshold:0.8"
)
os.environ.setdefault("CUDA_MODULE_LOADING", "LAZY")
os.environ.setdefault("CUDA_DEVICE_MAX_CONNECTIONS", "1")

import json
import sqlite3
import threading
import time
import uuid
from datetime import datetime
from PIL import Image
import torch
from flask import Flask, request, render_template_string, jsonify, send_file, redirect, url_for
from werkzeug.utils import secure_filename

# NOTE: use the multi-image capable pipeline
from diffusers import QwenImageEditPlusPipeline

# Configuration
UPLOAD_FOLDER = 'input'
OUTPUT_FOLDER = 'output'
DATABASE_FILE = 'qwenie.sqlite'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp'}

# Ensure directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # per-file 16MB

# Globals
pipeline = None
processing_thread = None
processing_active = True

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def init_database():
    """
    Create or migrate DB.
    New field: 'filenames' (JSON array of inputs). We keep legacy 'filename' for back-compat.
    """
    conn = sqlite3.connect(DATABASE_FILE)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS jobs (
            id TEXT PRIMARY KEY,
            filename TEXT, -- legacy single filename (kept for back-compat)
            filenames TEXT, -- NEW: JSON array of input filenames
            prompt TEXT NOT NULL,
            inference_steps INTEGER NOT NULL,
            status TEXT DEFAULT 'pending',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            completed_at TIMESTAMP,
            output_filename TEXT,
            error_message TEXT
        )
    ''')
    # If an older DB exists without 'filenames', add it
    cursor.execute("PRAGMA table_info(jobs)")
    cols = {row[1] for row in cursor.fetchall()}
    if 'filenames' not in cols:
        cursor.execute("ALTER TABLE jobs ADD COLUMN filenames TEXT")
    conn.commit()
    conn.close()

def load_pipeline():
    global pipeline
    print("Loading pipeline...")

    # Default to the official 2509 bf16 diffusers repo. Override with MODEL_ID to point at a 4-bit pack
    # (e.g. pre-2509: 'ovedrive/qwen-image-edit-4bit'). When a 2509 Q4 safetensors pack exists,
    # set MODEL_ID to that repo id and restart.
    model_id = os.environ.get("MODEL_ID", "ovedrive/Qwen-Image-Edit-2509-4bit")

    # Qwen-Image-Edit-Plus supports multi-image input
    pipeline = QwenImageEditPlusPipeline.from_pretrained(model_id, torch_dtype=torch.bfloat16)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipeline.to(device)
    pipeline.set_progress_bar_config(disable=None)
    print(f"Pipeline loaded from {model_id} on {device}")

def read_job_row(row):
    # row schema: id, filename, filenames, prompt, inference_steps, status, created_at, completed_at, output_filename, error_message
    return {
        'id': row[0],
        'filename': row[1],
        'filenames': row[2],
        'prompt': row[3],
        'inference_steps': row[4],
        'status': row[5],
        'created_at': row[6],
        'completed_at': row[7],
        'output_filename': row[8],
        'error_message': row[9],
    }

def load_images_from_job(job):
    """
    Returns a list[Image] (>=1).
    Supports legacy single 'filename' and new 'filenames' JSON list.
    """
    images = []
    if job['filenames']:
        try:
            flist = json.loads(job['filenames'])
            for fname in flist:
                path = os.path.join(UPLOAD_FOLDER, fname)
                if os.path.exists(path):
                    images.append(Image.open(path).convert("RGB"))
        except Exception:
            pass
    # Fallback to legacy single filename
    if not images and job['filename']:
        path = os.path.join(UPLOAD_FOLDER, job['filename'])
        if os.path.exists(path):
            images.append(Image.open(path).convert("RGB"))

    if not images:
        raise FileNotFoundError("No input images found for job")
    return images

def process_queue():
    global pipeline, processing_active

    while processing_active:
        conn = sqlite3.connect(DATABASE_FILE)
        cursor = conn.cursor()

        cursor.execute("SELECT * FROM jobs WHERE status = 'pending' ORDER BY created_at LIMIT 1")
        row = cursor.fetchone()

        if row:
            job = read_job_row(row)
            job_id = job['id']
            try:
                print(f"Processing job {job_id}: {job['prompt']}")
                cursor.execute("UPDATE jobs SET status = 'processing' WHERE id = ?", (job_id,))
                conn.commit()

                images = load_images_from_job(job)
                # Build inputs for Plus pipeline (accepts single PIL or list of PILs)
                inputs = {
                    "image": images if len(images) > 1 else images[0],
                    "prompt": job['prompt'],
                    "generator": torch.manual_seed(0),
                    "true_cfg_scale": 4.0,
                    "negative_prompt": " ",
                    "num_inference_steps": int(job['inference_steps']),
                    "guidance_scale": 1.0,
                    "num_images_per_prompt": 1,
                }

                with torch.inference_mode():
                    output = pipeline(**inputs)

                # Save output
                output_filename = f"{job_id}_output.png"
                output_path = os.path.join(OUTPUT_FOLDER, output_filename)
                output.images[0].save(output_path)

                cursor.execute("""
                    UPDATE jobs
                       SET status='completed',
                           completed_at=CURRENT_TIMESTAMP,
                           output_filename=?
                     WHERE id=?
                """, (output_filename, job_id))
                conn.commit()
                print(f"Job {job_id} completed")
            except Exception as e:
                err = str(e)
                print(f"Error processing job {job_id}: {err}")
                cursor.execute("""
                    UPDATE jobs
                       SET status='error',
                           error_message=?,
                           completed_at=CURRENT_TIMESTAMP
                     WHERE id=?
                """, (err, job_id))
                conn.commit()

        conn.close()
        time.sleep(1)

HTML_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
    <title>Qwen Image Edit (Multi-Image)</title>
    <style>
        body { font-family: Arial, sans-serif; max-width: 900px; margin: 0 auto; padding: 20px; }
        .form-group { margin-bottom: 15px; }
        label { display: block; margin-bottom: 5px; font-weight: bold; }
        input, textarea, button { width: 100%; padding: 8px; margin-bottom: 10px; }
        button { background-color: #007bff; color: white; border: none; cursor: pointer; }
        button:hover { background-color: #0056b3; }
        .job { border: 1px solid #ddd; padding: 15px; margin: 10px 0; border-radius: 5px; }
        .pending { background-color: #fff3cd; }
        .processing { background-color: #d1ecf1; }
        .completed { background-color: #d4edda; }
        .error { background-color: #f8d7da; }
        .download-btn { display: inline-block; background-color: #28a745; color: white; text-decoration: none; padding: 5px 10px; border-radius: 3px; }
        .refresh-btn { background-color: #6c757d; color: white; text-decoration: none; padding: 5px 10px; border-radius: 3px; display: inline-block; margin-bottom: 20px; }
        .small { font-size: 12px; color: #555; }
    </style>
</head>
<body>
    <h1>Qwen Image Edit</h1>
    <p class="small">Multi-image inputs supported (optional). Default model: <code>{{ model_id }}</code>. Override with <code>MODEL_ID</code> env var.</p>

    <h2>Submit New Job</h2>
    <form action="/upload" method="post" enctype="multipart/form-data">
        <div class="form-group">
            <label for="files">Select Image(s):</label>
            <input type="file" id="files" name="files" accept="image/*" multiple required>
            <span class="small">Tip: select up to ~3 for best VRAM usage.</span>
        </div>

        <div class="form-group">
            <label for="prompt">Prompt:</label>
            <textarea id="prompt" name="prompt" rows="3" required placeholder="Describe the edit you want..."></textarea>
        </div>

        <div class="form-group">
            <label for="inference_steps">Inference Steps:</label>
            <input type="number" id="inference_steps" name="inference_steps" value="20" min="5" max="75" required>
        </div>

        <button type="submit">Submit Job</button>
    </form>

    <hr>

    <h2>Jobs Queue</h2>
    <a href="/" class="refresh-btn">Refresh</a>

    <div id="jobs">
        {% for job in jobs %}
        <div class="job {{ job.status }}">
            <strong>Job ID:</strong> {{ job.id }}<br>
            <strong>Status:</strong> {{ job.status }}<br>
            <strong>Prompt:</strong> {{ job.prompt }}<br>
            <strong>Inference Steps:</strong> {{ job.inference_steps }}<br>
            <strong>Created:</strong> {{ job.created_at }}<br>
            {% if job.filenames %}
                <strong>Inputs:</strong> {{ job.filenames }}<br>
            {% elif job.filename %}
                <strong>Input:</strong> {{ job.filename }}<br>
            {% endif %}

            {% if job.status == 'completed' and job.output_filename %}
                <strong>Output:</strong>
                <a href="/download/{{ job.output_filename }}" class="download-btn">Download Result</a>
            {% elif job.status == 'error' %}
                <strong>Error:</strong> {{ job.error_message }}
            {% endif %}
        </div>
        {% endfor %}
    </div>
</body>
</html>
'''

@app.route('/')
def index():
    conn = sqlite3.connect(DATABASE_FILE)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM jobs ORDER BY created_at DESC")
    rows = cursor.fetchall()
    conn.close()

    jobs = [read_job_row(r) for r in rows]
    model_id = os.environ.get("MODEL_ID", "ovedrive/Qwen-Image-Edit-2509-4bit")
    return render_template_string(HTML_TEMPLATE, jobs=jobs, model_id=model_id)

@app.route('/upload', methods=['POST'])
def upload_file():
    # Accept single or multiple files from <input name="files" multiple>
    incoming = request.files.getlist('files')
    if not incoming:
        return redirect(url_for('index'))

    prompt = request.form.get('prompt', '').strip()
    inference_steps = int(request.form.get('inference_steps', 20))

    valid_files = [f for f in incoming if f and f.filename and allowed_file(f.filename)]
    if not valid_files or not prompt:
        return redirect(url_for('index'))

    job_id = str(uuid.uuid4())

    saved_names = []
    for f in valid_files:
        fname = secure_filename(f"{job_id}_{f.filename}")
        fpath = os.path.join(app.config['UPLOAD_FOLDER'], fname)
        f.save(fpath)
        saved_names.append(fname)

    # Insert job (store list in 'filenames' JSON). Also store first file in legacy 'filename'
    conn = sqlite3.connect(DATABASE_FILE)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO jobs (id, filename, filenames, prompt, inference_steps, status)
        VALUES (?, ?, ?, ?, ?, 'pending')
    """, (job_id, saved_names[0], json.dumps(saved_names), prompt, inference_steps))
    conn.commit()
    conn.close()

    print(f"New job submitted: {job_id} with {len(saved_names)} image(s)")
    return redirect(url_for('index'))

@app.route('/download/<filename>')
def download_file(filename):
    filepath = os.path.join(app.config['OUTPUT_FOLDER'], filename)
    if os.path.exists(filepath):
        return send_file(filepath, as_attachment=True)
    else:
        return "File not found", 404

def main():
    global processing_thread
    print("Initializing database...")
    init_database()

    print("Loading AI pipeline...")
    load_pipeline()

    print("Starting processing thread...")
    processing_thread = threading.Thread(target=process_queue, daemon=True)
    processing_thread.start()

    print("Starting web server on http://localhost:5000")
    app.run(host='0.0.0.0', port=5000, debug=False)

if __name__ == '__main__':
    main()
