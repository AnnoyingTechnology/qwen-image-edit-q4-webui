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
import sqlite3
import threading
import time
import uuid
from datetime import datetime
from PIL import Image
import torch
from flask import Flask, request, render_template_string, jsonify, send_file, redirect, url_for
from werkzeug.utils import secure_filename
from diffusers import QwenImageEditPipeline

# Configuration
UPLOAD_FOLDER = 'input'
OUTPUT_FOLDER = 'output'
DATABASE_FILE = 'qwenie.sqlite'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}

# Ensure directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Global variables
pipeline = None
processing_thread = None
processing_active = True

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def init_database():
    conn = sqlite3.connect(DATABASE_FILE)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS jobs (
            id TEXT PRIMARY KEY,
            filename TEXT NOT NULL,
            prompt TEXT NOT NULL,
            inference_steps INTEGER NOT NULL,
            status TEXT DEFAULT 'pending',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            completed_at TIMESTAMP,
            output_filename TEXT,
            error_message TEXT
        )
    ''')
    conn.commit()
    conn.close()

def load_pipeline():
    global pipeline
    print("Loading pipeline...")
    model_path = "ovedrive/qwen-image-edit-4bit"
    pipeline = QwenImageEditPipeline.from_pretrained(model_path, torch_dtype=torch.bfloat16)
    pipeline.to("cuda")
    pipeline.set_progress_bar_config(disable=None)
    print("Pipeline loaded successfully")

def process_queue():
    global pipeline, processing_active
    
    while processing_active:
        conn = sqlite3.connect(DATABASE_FILE)
        cursor = conn.cursor()
        
        # Get next pending job
        cursor.execute("SELECT * FROM jobs WHERE status = 'pending' ORDER BY created_at LIMIT 1")
        job = cursor.fetchone()
        
        if job:
            job_id, filename, prompt, inference_steps, status, created_at, completed_at, output_filename, error_message = job
            
            try:
                print(f"Processing job {job_id}: {prompt}")
                
                # Update status to processing
                cursor.execute("UPDATE jobs SET status = 'processing' WHERE id = ?", (job_id,))
                conn.commit()
                
                # Load and process image
                image_path = os.path.join(UPLOAD_FOLDER, filename)
                image = Image.open(image_path).convert("RGB")
                
                inputs = {
                    "image": image,
                    "prompt": prompt,
                    "generator": torch.manual_seed(0),
                    "true_cfg_scale": 4.0,
                    "negative_prompt": " ",
                    "num_inference_steps": inference_steps,
                }
                
                with torch.inference_mode():
                    output = pipeline(**inputs)
                
                # Save output
                output_filename = f"{job_id}_output.png"
                output_path = os.path.join(OUTPUT_FOLDER, output_filename)
                output_image = output.images[0]
                output_image.save(output_path)
                
                # Update job as completed
                cursor.execute("""
                    UPDATE jobs SET 
                        status = 'completed', 
                        completed_at = CURRENT_TIMESTAMP, 
                        output_filename = ? 
                    WHERE id = ?
                """, (output_filename, job_id))
                conn.commit()
                
                print(f"Job {job_id} completed successfully")
                
            except Exception as e:
                error_msg = str(e)
                print(f"Error processing job {job_id}: {error_msg}")
                
                cursor.execute("""
                    UPDATE jobs SET 
                        status = 'error', 
                        error_message = ?, 
                        completed_at = CURRENT_TIMESTAMP 
                    WHERE id = ?
                """, (error_msg, job_id))
                conn.commit()
        
        conn.close()
        time.sleep(1)  # Check every second

HTML_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
    <title>Qwen Image Edit</title>
    <style>
        body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
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
    </style>
</head>
<body>
    <h1>Qwen Image Edit</h1>
    
    <h2>Submit New Job</h2>
    <form action="/upload" method="post" enctype="multipart/form-data">
        <div class="form-group">
            <label for="file">Select Image:</label>
            <input type="file" id="file" name="file" accept="image/*" required>
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
    jobs = cursor.fetchall()
    conn.close()
    
    job_list = []
    for job in jobs:
        job_dict = {
            'id': job[0],
            'filename': job[1],
            'prompt': job[2],
            'inference_steps': job[3],
            'status': job[4],
            'created_at': job[5],
            'completed_at': job[6],
            'output_filename': job[7],
            'error_message': job[8]
        }
        job_list.append(job_dict)
    
    return render_template_string(HTML_TEMPLATE, jobs=job_list)

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(url_for('index'))
    
    file = request.files['file']
    prompt = request.form.get('prompt', '').strip()
    inference_steps = int(request.form.get('inference_steps', 50))
    
    if file.filename == '' or not prompt:
        return redirect(url_for('index'))
    
    if file and allowed_file(file.filename):
        # Generate unique job ID and filename
        job_id = str(uuid.uuid4())
        filename = secure_filename(f"{job_id}_{file.filename}")
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Add job to database
        conn = sqlite3.connect(DATABASE_FILE)
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO jobs (id, filename, prompt, inference_steps, status) 
            VALUES (?, ?, ?, ?, 'pending')
        """, (job_id, filename, prompt, inference_steps))
        conn.commit()
        conn.close()
        
        print(f"New job submitted: {job_id}")
    
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
