FROM python:3.9-trixie
LABEL authors="mariantasina"

RUN mkdir spam
WORKDIR spam
COPY requirements_transformers.txt /spam/requirements.txt

RUN pip install -r /spam/requirements.txt

COPY models/boltuix_bert_emotion ./models/boltuix_bert_emotion
COPY servers/emotion_transformer_server_api.py ./servers/emotion_transformer_server_api.py


CMD ["python", "servers/emotion_transformer_server_api.py"]