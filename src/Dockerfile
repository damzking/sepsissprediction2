# Use the official Python 3.10.9 image from the Docker Hub
FROM python:3.10.9

# Copy the requirements file into the /tmp directory in the Docker image
COPY requirements.txt /tmp/requirements.txt

# Install the Python packages specified in requirements.txt
# The timeout parameter is moved to the correct position
RUN python -m pip install --timeout 300 --no-cache-dir -r /tmp/requirements.txt

# Copy all files from the current directory into /app directory in the Docker image
COPY . /app

# Set the working directory to /app
WORKDIR /app

# Expose port 8000 for the application
EXPOSE 8000

# Command to run when starting the container
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
