FROM python:3.9-slim
LABEL authors="mariantasina"

RUN mkdir spam
WORKDIR spam
COPY ../../requirements_transformers.txt /spam/requirements.txt

RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch==2.3.1

RUN pip install -r /spam/requirements.txt

#COPY models/boltuix_bert_emotion ./models/boltuix_bert_emotion
COPY ../../UI/shiny_app.py ./UI/shiny_app.py
COPY client ./client/

CMD ["shiny","run","--host","0.0.0.0","--port","8080","UI/shiny_app.py"]