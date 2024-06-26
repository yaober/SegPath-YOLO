FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
RUN pip install gdown
RUN mkdir -p /app/ckpts/livecell/weights
RUN gdown 15-avlWF4LDdUEK5BNFfCb5a9j_BStt6G -O /app/ckpts/livecell/weights
RUN gdown 1owhHZQIj0FWDNRWsEKaTm9gcvQThA7Et -O /app/ckpts/livecell/weights

CMD ["python", "app_gradio.py"]
