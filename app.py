from flask import Flask, render_template, request, send_file, jsonify
import os
import logging
import sys
from werkzeug.utils import secure_filename
import cv2
import numpy as np
from pdf2image import convert_from_path
from PIL import Image
import io
import tempfile
import shutil

# Configure logging to show everything
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024  # 5MB max file size

# Use system temp directory
app.config['UPLOAD_FOLDER'] = tempfile.gettempdir()

def process_image(image_path):
    """Process a single image file."""
    try:
        logger.info(f"Processing image: {image_path}")
        
        # Read image
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError("Failed to read image")
        
        # Basic thresholding
        _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Save processed image
        cv2.imwrite(image_path, binary)
        logger.info("Image processing completed")
        return True
    except Exception as e:
        logger.error(f"Error processing image: {str(e)}")
        raise

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        logger.info("Starting file upload process")
        
        if 'file' not in request.files:
            logger.error("No file part in request")
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        if file.filename == '':
            logger.error("No selected file")
            return jsonify({'error': 'No file selected'}), 400
        
        if not file.filename.lower().endswith('.pdf'):
            logger.error("Invalid file type")
            return jsonify({'error': 'Only PDF files are allowed'}), 400
        
        # Create temporary directory
        temp_dir = tempfile.mkdtemp()
        input_path = os.path.join(temp_dir, secure_filename(file.filename))
        output_path = os.path.join(temp_dir, 'output.pdf')
        
        try:
            logger.info(f"Saving uploaded file to {input_path}")
            file.save(input_path)
            
            # Convert first page of PDF to image
            logger.info("Converting PDF to image")
            images = convert_from_path(
                input_path,
                first_page=1,
                last_page=1,
                dpi=200,
                fmt='jpeg',
                output_folder=temp_dir
            )
            
            if not images:
                raise ValueError("Failed to convert PDF to image")
            
            # Process the image
            image_path = os.path.join(temp_dir, 'out_1.jpg')
            process_image(image_path)
            
            # Convert back to PDF
            logger.info("Converting processed image back to PDF")
            img = Image.open(image_path)
            img.save(output_path, 'PDF', resolution=100.0)
            
            logger.info("Sending processed file")
            return send_file(
                output_path,
                as_attachment=True,
                download_name=f"enhanced_{file.filename}"
            )
            
        except Exception as e:
            logger.error(f"Processing error: {str(e)}")
            return jsonify({'error': str(e)}), 500
        
        finally:
            try:
                logger.info(f"Cleaning up temporary directory: {temp_dir}")
                shutil.rmtree(temp_dir, ignore_errors=True)
            except Exception as e:
                logger.error(f"Cleanup error: {str(e)}")
    
    except Exception as e:
        logger.error(f"Upload error: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
