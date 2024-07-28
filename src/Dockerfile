# Use the official Python 3.9.6 image as the base image
FROM python:3.9.6


# Set the working directory inside the container to /app
WORKDIR /app


# Update package list, install curl, and install Rust
RUN apt-get update && apt-get install -y curl \
    && curl https://sh.rustup.rs -sSf | sh -s -- -y \
    && . $HOME/.cargo/env \
    && export PATH="$HOME/.cargo/bin:$PATH"

# Set the PATH environment variable to include Rust binaries
ENV PATH="/root/.cargo/bin:${PATH}"

# Copy the requirements.txt file into the /tmp directory inside the container
COPY requirements.txt /tmp/

# Upgrade pip and install the required Python packages listed in requirements.txt
RUN python -m pip install --upgrade pip && python -m pip install --timeout 300 --no-cache-dir -r /tmp/requirements.txt

# Copy the entire content of the current directory (where the Dockerfile is located) into the /app directory inside the container
COPY . /app

# Expose port 8000 to allow communication to/from the container
EXPOSE 8000

# Specify the command to run the FastAPI application using uvicorn when the container starts
CMD ["uvicorn", "myApp:app", "--host", "0.0.0.0", "--port", "8000"]