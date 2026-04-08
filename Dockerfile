# Use a slim Python 3.11 image for better compatibility
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy requirements file first to leverage Docker cache
COPY requirements.txt .

# Install dependencies without cache
RUN pip install --no-cache-dir -r requirements.txt

# Install FastAPI and Uvicorn explicitly
RUN pip install fastapi uvicorn python-dotenv openai

# Copy the rest of your project files
COPY . .

# Set environment variable for Python unbuffered output
ENV PYTHONUNBUFFERED=1

# Expose port 8080 (or any port your FastAPI app uses)
EXPOSE 8080

# Command to run FastAPI server
CMD ["uvicorn", "inference:app", "--host", "0.0.0.0", "--port", "8080", "--reload"]