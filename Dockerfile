FROM python:3.10

#input container mount
RUN mkdir /input
# output container mount
RUN mkdir /output
RUN mkdir /app
COPY pyproject.toml poetry.lock /app/
# we are now getting poetry on the docker image
# we cant be there to hit y/n on settings
# ENV DEBIAN_FRONTEND=noninteractive
ENV POETRY_VIRTUALENVS_CREATE=false
RUN pip install 'poetry==1.8.3'
#RUN pip install poetry && poetry install --only main --no-root --no-directory #OLD VERSION
ENV PATH="/root/.local/bin:/app/:${PATH}"
WORKDIR /app
RUN poetry install --no-interaction --no-ansi
# now copy in the mechanistic model code and config code
# we will upload the experiment itself into the cloud and refer to from /input
COPY ./mechanistic_model/ /app/mechanistic_model
COPY ./config/config.py /app/config/config.py
COPY ./model_odes/seip_model.py /app/model_odes/seip_model.py
# COPY ./utils.py /app/utils.py
