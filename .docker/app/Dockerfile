FROM python:3.13-slim

WORKDIR /app

COPY .docker/app/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/
COPY ui/ ui/

EXPOSE 8080

CMD ["uvicorn", "src.vllm.app:app", "--host", "0.0.0.0", "--port", "8080"]
