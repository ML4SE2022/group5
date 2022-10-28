FROM python:3.10

COPY requirements.txt .

RUN pip install -r requirements.txt

COPY dataset/ dataset/
COPY type-model/ type-model/
COPY src/ .
COPY models/ models/

ENTRYPOINT [ "python", "train.py" ]
