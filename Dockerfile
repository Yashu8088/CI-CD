FROM python:3.10-slim

WORKDIR /app

# Copy and install dependencies first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Train model during image build
RUN python train.py

# Expose Flask port
EXPOSE 5000

# Run Flask app
CMD ["python", "app.py"]
