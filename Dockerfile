# Stage 1: `python-base` sets up all our shared environment variables.
FROM python:3.9-slim as python-base

ENV PYTHONUNBUFFERED=1 \
    # prevents python creating .pyc files
    PYTHONDONTWRITEBYTECODE=1 \
    \
    # PIP
    PIP_NO_CACHE_DIR=off \
    PIP_DISABLE_PIP_VERSION_CHECK=on \
    PIP_DEFAULT_TIMEOUT=100 \
    \
    # POETRY
    # https://python-poetry.org/docs/configuration/#using-environment-variables
    POETRY_VERSION=1.1.13 \
    # Install poetry to this location
    POETRY_HOME="/opt/poetry" \
    # make poetry create the virtual environment in the project's root
    # it gets named `.venv`
    POETRY_VIRTUALENVS_IN_PROJECT=true \
    # do not ask any interactive question
    POETRY_NO_INTERACTION=1 \
    # poetry virtual env location
    VENV_PATH="/.venv"

# Add POETRY_HOME to path, and VENV_PATH (poetry virtual env location) to path
ENV PATH="$POETRY_HOME/bin:$VENV_PATH/bin:$PATH"


# Stage 2: `builder-base` stage is used to build deps + create our virtual environment. Cuda already installed on pytorch wheel
FROM python-base as builder-base

RUN apt-get update \
    && apt-get install --no-install-recommends -y \
        # deps for installing poetry
        curl \
        # deps for building python deps
        build-essential

# install poetry - respects $POETRY_VERSION & $POETRY_HOME
RUN curl -sSL https://raw.githubusercontent.com/sdispater/poetry/master/get-poetry.py | python3

# Copy requirements
COPY poetry.lock pyproject.toml ./

# Marked as unsafe on poetry < 1.20: cannnot be added directly into pyproject.toml. Needed for pytorch (bug with last v)
RUN poetry run pip install 'setuptools==59.5.0'

RUN poetry install --no-root --no-dev  # No dev packages and project's package

# Install torch (use poetry cause conflicts)
RUN poe force-cuda11


# Stage 3: `production` image used for runtime
FROM python-base as production

COPY --from=builder-base $VENV_PATH $VENV_PATH

# Install project
COPY src/image_manager/super_resolution /super_resolution

# Pull trained models from google drive with DVC. You should have a conf file with credentials in conf/
COPY conf /conf
COPY models.dvc ./
COPY .dvc/config /.dvc/config
RUN dvc config core.no_scm true  # Tells DVC to no look for a git repo.
RUN dvc pull models

# Copy demo
COPY api/app /app

# Add project root to path (not to $PATH !)
ENV PYTHONPATH="${PYTHONPATH}:/super_resolution"

WORKDIR /app

# Since Flask apps listen to port 5000  by default, we expose it
EXPOSE 5000
