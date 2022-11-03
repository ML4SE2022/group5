FROM python:3.10

COPY requirements.txt .

RUN pip install -r requirements.txt

COPY dataset/ dataset/
# COPY models/ models/
COPY src/ .

ENTRYPOINT [ "python", "train.py" ]
