FROM python:3.11-slim

WORKDIR /app
COPY . .

RUN pip install --no-cache-dir -r bun_forecast_demo/requirements.txt

ENV PYTHONPATH=/app
EXPOSE 7860

CMD ["uvicorn", "bun_forecast_demo.app:app", "--host", "0.0.0.0", "--port", "7860"]
