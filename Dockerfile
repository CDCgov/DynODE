FROM python:3.10

#input container mount
RUN mkdir /input
# output container mount
RUN mkdir /output
# where the code is going to be placed
RUN mkdir /app
# copy in the poetry files to install libraries
COPY pyproject.toml poetry.lock /app/
# disable venv creation as docker is inheritly isolated
# the point of a venv is to isolate packages from other repos
ENV POETRY_VIRTUALENVS_CREATE=false
# lock poetry version to 1.8.3 which is known to work
RUN pip install 'poetry==1.8.3'
# add poetry location to path so we can call `poetry install`
# also add `app` so we can reference imports in files easily
ENV PATH="/root/.local/bin:/app/:${PATH}"
# set workdir to /app for installing packages
WORKDIR /app
# turn off interaction since we cant type `yes` on the prompts in docker build
RUN poetry install --no-interaction --no-ansi
# now copy in the mechanistic model code and config code
# we will upload the experiment itself into the cloud and refer to from /input
COPY ./mechanistic_model/ /app/mechanistic_model
COPY ./config/config.py /app/config/config.py
COPY ./model_odes/seip_model.py /app/model_odes/seip_model.py
