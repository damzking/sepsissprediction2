FROM python:3.9.6

WORKDIR /app

#COPY requirements.txt /tmp/requirements.txt

RUN apt-get update && apt-get install -y curl \
    && curl https://sh.rustup.rs -sSf | sh -s -- -y \
    && . $HOME/.cargo/env \
    && export PATH="$HOME/.cargo/bin:$PATH"

ENV PATH="/root/.cargo/bin:${PATH}"


COPY requirements.txt /tmp/

RUN python -m pip install --upgrade pip && python -m pip install --timeout 300 --no-cache-dir -r /tmp/requirements.txt

COPY . /app

#WORKDIR /app

EXPOSE 8501

CMD ["streamlit", "run", "main.py", "--server.port=8501"]