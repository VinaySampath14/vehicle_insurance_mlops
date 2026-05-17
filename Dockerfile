FROM python:3.10-slim-buster

WORKDIR /app

COPY requirements.txt setup.py ./

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8080

CMD ["python", "app.py"]
