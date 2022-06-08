# Stage 1: Builder/Compiler
FROM pytorch/pytorch:1.10.0-cuda11.3-cudnn8-runtime

WORKDIR /app

# Copy requirements
COPY pyproject.toml /app

# Set environment variables
ENV PYTHONPATH "${PYTHONPATH}:/src"
# Python output is sent straight to terminal without being first buffered
ENV PYTHONUNBUFFERED 1

RUN pip install poetry==1.1.13
RUN poetry config virtualenvs.create false  # No need to create a virtual env (already inside a docker image)
RUN poetry install --no-root --no-dev  # No dev packages and project's package
RUN pip install torchvision==0.11.0+cu113 -f https://download.pytorch.org/whl/torch_stable.html --no-deps  # Compatible with torch based image and cuda

# Install project
COPY src/image_manager/super_resolution /src

# Copy trained models
COPY models /models

# Copy demo
COPY api/app /app

#TODO add models
#CMD ["gunicorn", "--bind", "0.0.0.0:5000", "server:app"]
#ENTRYPOINT ["gunicorn", "-b", "0.0.0.0:8000", "--access-logfile", "-", "--error-logfile", "-", "--timeout", "120"]
