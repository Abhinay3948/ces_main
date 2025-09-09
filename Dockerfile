# Use official Python image
FROM python:3.10-slim

# Install system dependencies (Poppler + Tesseract + fonts + build tools)
RUN apt-get update && apt-get install -y \
    poppler-utils \
    tesseract-ocr \
    libtesseract-dev \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set workdir
WORKDIR /app

# Copy requirements and install Python deps
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy all files into container
COPY . .

# Expose port (Render will map $PORT to this)
EXPOSE 8080

# Start Streamlit app
CMD ["streamlit", "run", "app.py", "--server.port=8080", "--server.address=0.0.0.0"]
