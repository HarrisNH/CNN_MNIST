FROM python:3.11-slim

RUN apt-get update && apt-get install 
RUN mkdir -p /models /reports/figures

COPY requirements.txt requirements.txt
COPY pyproject.toml pyproject.toml
COPY src/cnn_mnist/ src/cnn_mnist/
COPY data/ data/

WORKDIR /
RUN --mount=type=cache,target=~/pip/.cache pip install -r requirements.txt --no-cache-dir
RUN pip install . --no-deps --no-cache-dir

ENTRYPOINT ["python", "-u", "src/cnn_mnist/train.py"]