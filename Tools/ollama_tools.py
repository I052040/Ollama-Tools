# pip install ollama , flask

import subprocess
import sys
import socket
import os
try:
    import ollama
except ImportError:
    print("Warning: Ollama Service Not Found")
    ollama = None
from flask import Flask, render_template, request, jsonify, redirect, url_for
from werkzeug.utils import secure_filename
import threading
import time
import tempfile
import uuid
from datetime import datetime

# Constants
RESPONSE_CONTENT_OLLAMA_FILE = "content_out @ollama.md"
TEMP_IMAGE_FILE = "temp.jpg"
OLLAMA_HOST = "localhost"  # Default Ollama host
OLLAMA_PORT = 11434  # Default Ollama port
OLLAMA_PROMPT = "Describe Image ,Extract text，Don't translate text:"

# Dictionary to store ongoing tasks
tasks = {}

def get_ollama_models():
    """Retrieves a list of available Ollama models using 'ollama list'.
    Handles potential errors gracefully and provides informative messages.
    """
    try:
        output = subprocess.check_output(['ollama', 'list'], text=True, stderr=subprocess.PIPE)
        model_names = [
            detail.split()[0]
            for detail in output.split('\n')
            if detail and detail.split()[0] != "NAME"  # Prevent empty strings and header from being processed
        ]
        return model_names
    except subprocess.CalledProcessError as e:
        print(f"Error listing models: {e}")
        print(f"Ollama output: {e.output}")
        return []
    except FileNotFoundError:
        print("Error: Ollama command not found. Ensure Ollama is installed and in your system's PATH.")
        return []
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return []


def get_response(system_message, user_input, llm_model, images=None):
    """Retrieves a response from the specified Ollama model.
    Includes error handling for API requests.
    """
    messages = [{'role': 'system', 'content': system_message}] if system_message else []
    messages.append({'role': 'user', 'content': user_input})
    if images:
        messages[-1]['images'] = images

    try:
        response = ollama.chat(model=llm_model, messages=messages)
        return response['message']['content']
    except Exception as e:
        print(f"Error getting response from Ollama: {e}")
        return f"Error: Could not get response from {llm_model}. Check the console for details."


def save_response(response_content, filename=RESPONSE_CONTENT_OLLAMA_FILE):
    """Appends the chatbot's response to a file.
    Includes basic error handling.
    """
    try:
        with open(filename, "w") as file:
            file.write(response_content + "\n")
    except IOError as e:
        print(f"Error saving response to file: {e}")


def is_port_open(host, port):
    """Check if a port is open on the specified host.
    Improved error handling and reporting.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(1)  # Set a timeout to prevent indefinite blocking
        try:
            s.connect((host, port))
            return True
        except socket.timeout:
            print(f"Connection to {host}:{port} timed out.")
            return False
        except socket.error as e:
            print(f"Connection to {host}:{port} failed: {e}")
            return False
        except Exception as e:
            print(f"An unexpected error occurred while checking port: {e}")
            return False


def run_ollama_query(task_id, system_message, user_input, selected_model, images=None):
    """Runs the Ollama query in a background thread and updates the task status."""
    global tasks
    tasks[task_id] = {'status': 'running', 'progress': 0, 'result': None, 'error': None}
    
    try:
        # Simulate progress updates
        for progress in range(0, 100, 10):
            tasks[task_id]['progress'] = progress
            time.sleep(0.5)  # Reduced sleep time for faster testing

        response = get_response(system_message, user_input, selected_model, images)
        tasks[task_id]['progress'] = 100
        tasks[task_id]['status'] = 'completed'
        tasks[task_id]['result'] = response
    except Exception as e:
        print(f"Error in background task: {e}")
        tasks[task_id]['status'] = 'error'
        tasks[task_id]['error'] = str(e)
        tasks[task_id]['progress'] = 0


app = Flask(__name__)

# Configure upload folder
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def index():
    models = get_ollama_models()
    return render_template('ollama_tools.html', models=models)


@app.route('/api/models', methods=['GET'])
def api_models():
    models = get_ollama_models()
    return jsonify(models)


@app.route('/api/prompt', methods=['POST'])
def api_prompt():
    global tasks
    data = request.get_json()
    
    system_message = data.get('system_message', '')
    user_input = data.get('user_input', '')
    selected_model = data.get('selected_model', '')
    
    if not selected_model or not user_input:
        return jsonify({'error': 'Please select a model and enter a question.'}), 400
    
    # Generate a unique task ID
    task_id = str(uuid.uuid4())
    
    # Start the background task
    thread = threading.Thread(
        target=run_ollama_query,
        args=(task_id, system_message, user_input, selected_model)
    )
    thread.start()
    
    return jsonify({'task_id': task_id})


@app.route('/api/vision', methods=['POST'])
def api_vision():
    global tasks
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided.'}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No image selected.'}), 400
        
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type. Only image files (png, jpg, jpeg, gif, bmp) are supported.'}), 400
    
    # Save the uploaded file temporarily
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
    
    data = request.form
    selected_model = data.get('selected_model', '')
    
    if not selected_model:
        return jsonify({'error': 'Please select a model.'}), 400
    
    # Generate a unique task ID
    task_id = str(uuid.uuid4())
    
    # Start the background task
    thread = threading.Thread(
        target=run_ollama_query,
        args=(task_id, "", OLLAMA_PROMPT, [filepath])
    )
    thread.start()
    
    return jsonify({'task_id': task_id})


@app.route('/api/task_status/<task_id>')
def api_task_status(task_id):
    if task_id in tasks:
        return jsonify(tasks[task_id])
    else:
        return jsonify({'error': 'Task not found'}), 404


@app.route('/api/markdown', methods=['POST'])
def api_markdown():
    data = request.get_json()
    input_text = data.get('input_text', '')
    selected_model = data.get('selected_model', 'llama3.2')  # Default to a common model

    if not input_text:
        return jsonify({'error': 'No input text provided.'}), 400

    # Get response from Ollama model to convert text to Markdown
    try:
        response = get_response("", f"Convert the following text to GitHub Flavored Markdown (GFM) format. Preserve the structure and content as much as possible: {input_text}", selected_model)
        return jsonify({'result': response})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/save_markdown', methods=['POST'])
def api_save_markdown():
    data = request.get_json()
    content = data.get('content', '')
    
    if not content:
        return jsonify({'error': 'No content provided to save.'}), 400
    
    try:
        with open('output.md', 'w', encoding='utf-8') as f:
            f.write(content)
        return jsonify({'status': 'success', 'message': 'Markdown saved to output.md'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/markdown_editor')
def markdown_editor():
    models = get_ollama_models()
    return render_template('ollama_tools.html', models=models)


@app.route('/api/reset', methods=['POST'])
def api_reset():
    # Clear all tasks
    global tasks
    tasks = {}
    return jsonify({'status': 'ok'})


def find_available_port(starting_port=5000):
    """Finds an available port starting from the specified port number.
    Returns the first available port.
    """
    port = starting_port
    while True:
        if not is_port_open('0.0.0.0', port):
            return port
        print(f"Port {port} is occupied, checking next port...")
        port += 1


if __name__ == "__main__":
    if not is_port_open(OLLAMA_HOST, OLLAMA_PORT):
        print(f"Warning: Ollama does not seem to be running. Please ensure Ollama is running and accessible on {OLLAMA_HOST}:{OLLAMA_PORT}.")
    
    # Find an available port starting from 5000
    available_port = find_available_port(5000)
    print(f"Starting server on port {available_port}")
    
    app.run(debug=True, host='0.0.0.0', port=available_port)
