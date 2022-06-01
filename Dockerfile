FROM python:3.10.4-slim

WORKDIR /app

COPY ./src /tmp/src

# Set environment variables
ENV PYTHONUNBUFFERED 1

EXPOSE 5000

# Install project
COPY ./README.md /tmp/README.md
COPY ./src /tmp/src

# Copy project
COPY api /app