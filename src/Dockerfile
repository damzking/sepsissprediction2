FROM python:3.12.3

WORKDIR /app

RUN apt-get update && apt-get install -y curl \
    && curl https://sh.rustup.rs -sSf | sh -s -- -y \
    && . $HOME/.cargo/env \
    && export PATH="$HOME/.cargo/bin:$PATH"


ENV PATH="/root/.cargo/bin:${PATH}"


COPY requirements.txt /tmp/


RUN python -m pip install --upgrade pip && python -m pip install --timeout 300 --no-cache-dir -r /tmp/requirements.txt


COPY . /app


EXPOSE 8000


CMD ["uvicorn", "myApp:app", "--host", "0.0.0.0", "--port", "8000"]