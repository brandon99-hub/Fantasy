FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY backend/requirements.txt backend/pyproject.toml ./

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend code
COPY backend/ ./backend/

# Create necessary directories
RUN mkdir -p backend/data backend/logs

# Expose port
EXPOSE 8000

# Run the application
CMD ["uvicorn", "backend.src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]

