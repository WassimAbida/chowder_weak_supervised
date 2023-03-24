# ==========================================
# Base image
# ==========================================
FROM python:3.9 as base

RUN apt-get update && \
    apt-get install -y libsm6 libxrender1 libxext6 libgl1-mesa-glx && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Prepare for install python package using poetry
ENV PATH="$POETRY_HOME/bin:$PATH"

WORKDIR /opt/workspace

# Virtual env configuration
ENV VIRTUAL_ENV=./venv
RUN python -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

RUN python -m pip install --upgrade pip
RUN python -m pip install poetry

COPY pyproject.toml /opt/workspace/pyproject.toml
COPY poetry.lock /opt/workspace/poetry.lock
COPY chowder_weak_supervised /opt/workspace/chowder_weak_supervised
COPY app.sh /opt/workspace/app.sh
RUN chmod -R 777 /opt/workspace/app.sh

# Install project
RUN poetry config virtualenvs.create false \
&& poetry config virtualenvs.in-project false \
&& poetry install

ENTRYPOINT ["./app.sh"]
