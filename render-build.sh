

echo "Installing system dependencies..."
apt-get update
apt-get install -y tesseract-ocr tesseract-ocr-eng
apt-get install -y libgl1-mesa-glx libglib2.0-0 poppler-utils

echo "Installing Python dependencies..."
pip install -r requirements.txt

echo "Verifying installations..."
tesseract --version
python -c "import PyPDF2, pdfplumber, fitz; print('PDF libraries installed successfully')"

echo "Build completed successfully!"
