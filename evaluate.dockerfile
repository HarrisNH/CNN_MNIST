# Use the Python slim image as the base
FROM python:3.11-slim

# Install any necessary dependencies
RUN apt-get update && apt-get install 

# Set up directories for the application
RUN mkdir -p /models /data 

# Copy the requirements and install them
COPY requirements.txt /requirements.txt
COPY pyproject.toml pyproject.toml
COPY src/cnn_mnist/ src/cnn_mnist/
COPY data/ data/

WORKDIR /
RUN --mount=type=cache,target=~/pip/.cache pip install -r requirements.txt --no-cache-dir
RUN pip install . --no-deps --no-cache-dir

# Copy the prediction script and other project files
COPY models/model.pt /models/model.pt

# Set the entry point to run the evaluation script
ENTRYPOINT ["python", "-u", "src/cnn_mnist/evaluate.py"]