FROM python:3.10-slim

# Set environment variables to prevent Python from writing .pyc files
# and to keep stdout unbuffered.
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install basic sys requirements if needed by some ML packages (e.g., standard build tools)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file into the container
COPY requirements.txt .

# Install dependencies from requirements.txt
# (It might take a while to download PyTorch and transformers)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the application source code
COPY code/ ./code/

# Set the working directory to the code directory so scripts run easily
WORKDIR /app/code

# Keep container running interactively by default, or run a specific script.
CMD ["/bin/bash"]
