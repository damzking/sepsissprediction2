FROM python:3.9.6

# Set the working directory to /app
WORKDIR /app

# Install curl and Rust
RUN apt-get update && apt-get install -y curl \
    && curl https://sh.rustup.rs -sSf | sh -s -- -y \
    && . $HOME/.cargo/env \
    && export PATH="$HOME/.cargo/bin:$PATH"


# Add Rust binaries to the PATH environment variable
ENV PATH="/root/.cargo/bin:${PATH}"


# Copy requirements.txt to the /tmp directory
COPY requirements.txt /tmp/

# Upgrade pip and install the dependencies from requirements.txt
RUN python -m pip install --upgrade pip && python -m pip install --timeout 300 --no-cache-dir -r /tmp/requirements.txt

# Copy the entire content of the current directory to /app
COPY . /app


# Expose port 8501 for the Streamlit application
EXPOSE 8501


# Command to run the Streamlit application
CMD ["streamlit", "run", "main.py", "--server.port=8501"]