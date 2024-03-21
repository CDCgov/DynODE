FROM python:3.10

COPY . /app
#input container mount
RUN mkdir /input
# ADD /input /input
# output container mount
RUN mkdir /output
# ADD /output /output
# change cwd to /app
WORKDIR /app
RUN pip install -r requirements.txt
