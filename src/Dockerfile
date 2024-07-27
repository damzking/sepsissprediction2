FROM python:3.10.9

COPY  requirements.txt /tmp/requirements.txt

RUN Python -m pip install -- timeout 300000 requirements -r /tmp/requirements.txt

COPY . /app

WORKDIR /app

EXPOSE 8000

CMD [ "uvicorn" , "api:app", '--host', "0.0 0.0", "-port","8000" ]
