from flask import Flask, render_template, request, send_file
import os
from werkzeug.utils import secure_filename
import cv2
import numpy as np
from pdf2image import convert_from_path
from PIL import Image
import io
from PyPDF2 import PdfWriter, PdfReader

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['PROCESSED_FOLDER'] = 'processed'

# Ensure upload and processed directories exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['PROCESSED_FOLDER'], exist_ok=True)

def process_image(image):
    # Convert to grayscale
    gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    
    # Apply adaptive thresholding
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 15
    )
    
    # Denoise
    denoised = cv2.fastNlMeansDenoising(binary)
    
    return Image.fromarray(denoised)

def process_pdf(input_path, output_path):
    # Convert PDF to images
    images = convert_from_path(input_path)
    
    # Process each page
    processed_images = []
    for image in images:
        processed_image = process_image(image)
        processed_images.append(processed_image)
    
    # Save processed images as PDF
    processed_images[0].save(
        output_path,
        save_all=True,
        append_images=processed_images[1:],
        format='PDF'
    )

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return 'No file uploaded', 400
    
    file = request.files['file']
    if file.filename == '':
        return 'No file selected', 400
    
    if not file.filename.lower().endswith('.pdf'):
        return 'Only PDF files are allowed', 400
    
    filename = secure_filename(file.filename)
    input_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    output_path = os.path.join(app.config['PROCESSED_FOLDER'], f'processed_{filename}')
    
    file.save(input_path)
    
    try:
        process_pdf(input_path, output_path)
        return send_file(output_path, as_attachment=True)
    except Exception as e:
        return str(e), 500
    finally:
        # Clean up
        if os.path.exists(input_path):
            os.remove(input_path)
        if os.path.exists(output_path):
            os.remove(output_path)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
