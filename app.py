from flask import Flask, render_template, request, send_file, jsonify
import os
import logging
from werkzeug.utils import secure_filename
import cv2
import numpy as np
from pdf2image import convert_from_path
from PIL import Image
import io
from PyPDF2 import PdfWriter, PdfReader
import time

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max file size
app.config['UPLOAD_FOLDER'] = '/tmp/uploads'
app.config['PROCESSED_FOLDER'] = '/tmp/processed'

# Ensure upload and processed directories exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['PROCESSED_FOLDER'], exist_ok=True)

def process_image(image):
    try:
        # Convert to grayscale
        gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
        
        # Apply adaptive thresholding with more conservative parameters
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 10
        )
        
        # Light denoising
        denoised = cv2.fastNlMeansDenoising(binary, None, 10, 7, 21)
        
        return Image.fromarray(denoised)
    except Exception as e:
        logger.error(f"Error in process_image: {str(e)}")
        raise

def process_pdf(input_path, output_path):
    try:
        logger.info(f"Processing PDF: {input_path}")
        
        # Convert only first page as a test
        images = convert_from_path(input_path, first_page=1, last_page=1)
        logger.info(f"Converted first page of PDF")
        
        if not images:
            raise Exception("Failed to convert PDF to images")
        
        # Process the first page
        processed_image = process_image(images[0])
        
        # Save as PDF
        processed_image.save(output_path, format='PDF')
        logger.info(f"Saved processed PDF to {output_path}")
        
        return True
    except Exception as e:
        logger.error(f"Error in process_pdf: {str(e)}")
        raise

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        if 'file' not in request.files:
            logger.error('No file part in request')
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        if file.filename == '':
            logger.error('No selected file')
            return jsonify({'error': 'No file selected'}), 400
        
        if not file.filename.lower().endswith('.pdf'):
            logger.error('Invalid file type')
            return jsonify({'error': 'Only PDF files are allowed'}), 400
        
        # Generate unique filename
        timestamp = int(time.time())
        filename = f"{timestamp}_{secure_filename(file.filename)}"
        input_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        output_path = os.path.join(app.config['PROCESSED_FOLDER'], f'processed_{filename}')
        
        logger.info(f"Saving uploaded file to {input_path}")
        file.save(input_path)
        
        try:
            process_pdf(input_path, output_path)
            logger.info("File processed successfully")
            return send_file(
                output_path,
                as_attachment=True,
                download_name=f"enhanced_{os.path.basename(file.filename)}"
            )
        except Exception as e:
            logger.error(f"Processing error: {str(e)}")
            return jsonify({'error': f"Error processing file: {str(e)}"}), 500
        finally:
            # Clean up
            try:
                if os.path.exists(input_path):
                    os.remove(input_path)
                if os.path.exists(output_path):
                    os.remove(output_path)
            except Exception as e:
                logger.error(f"Cleanup error: {str(e)}")
    
    except Exception as e:
        logger.error(f"Upload error: {str(e)}")
        return jsonify({'error': f"Upload error: {str(e)}"}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
