# Use official Python base image
FROM python:3

# Set environment variables
ENV PYTHONFAULTHANDLER=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONHASHSEED=random \
    PIP_NO_CACHE_DIR=off \
    PIP_DISABLE_PIP_VERSION_CHECK=on \
    PIP_DEFAULT_TIMEOUT=100 \
    POETRY_VERSION=1.5.0

# Install dependencies
RUN pip install --upgrade pip
RUN pip install paramiko cryptography

# Copy SSH server code
COPY sshd.py /app/sshd.py

# Create and change directory
WORKDIR /app

# Expose SSH port
EXPOSE 22

# Command to run the SSH server
CMD ["python", "sshd.py"]