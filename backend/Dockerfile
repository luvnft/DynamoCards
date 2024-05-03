FROM python:3.9-slim AS backend
ENV PYTHONDONTWRITEBYTECOODE 1
ENV PYTHONBUFFERED 1
WORKDIR /app/backend

COPY requirements.txt /app/backend/
RUN python -m venv env && \
    .env/bin/activate && \
    pip install --no-cache-dir -r requirements.txt

COPY ./app/backend

EXPOSE 8008

CMD .env/bin/activate && uvicorn main:app -host 0.0.0.0 --port 8000