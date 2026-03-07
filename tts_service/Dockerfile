FROM python:3.10

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY tts_server.py .

CMD ["uvicorn","tts_server:app","--host","0.0.0.0","--port","8001"]
