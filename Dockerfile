FROM python:3.10

COPY . /app
WORKDIR /app
RUN pip install -r requirements.txt

CMD ["python", "example_end_to_end_run.py"]