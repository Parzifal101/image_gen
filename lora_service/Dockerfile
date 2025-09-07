# Dockerfile for lora_service/ (create this file in lora_service directory)
# Assuming it's a simple FastAPI service without GPU requirements for now (as per "forget about training code")

FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Copy requirements file (assume you have requirements.txt in lora_service)
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY . .

# Expose the FastAPI port
EXPOSE 8000

# Command to run the FastAPI app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]