FROM python:3.10

COPY . /app
#input container mount
RUN mkdir /input
# output container mount
RUN mkdir /output
WORKDIR /app
RUN pip install -r requirements.txt
