# syntax=docker/dockerfile:1.7
FROM python:3.12-slim

WORKDIR /app

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_INPUT=1

COPY requirements.txt ./
RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install -r requirements.txt

COPY pyproject.toml README.md ./
COPY src ./src
RUN python -m pip install --no-deps --no-build-isolation .

COPY policy.yaml ./policy.yaml
RUN mkdir -p /app/secrets /app/data

ENTRYPOINT ["openmailsweep"]
