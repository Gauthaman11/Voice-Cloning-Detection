# Use official lightweight Python image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies (required for librosa/soundfile)
RUN apt-get update && apt-get install -y \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Expose the port the app runs on
EXPOSE 5000

# Command to run the application with Gunicorn (Production WSGI Server)
# We use 1 worker and 4 threads for lightweight edge deployment
CMD ["gunicorn", "-w", "1", "--threads", "4", "-b", "0.0.0.0:5001", "api.app:app"]