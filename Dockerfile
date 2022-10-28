FROM python:3.10

COPY requirements.txt .

RUN pip install -r requirements.txt

COPY 50k_types/ 50k_types/
COPY type-model/ type-model/
COPY src/ .
COPY models models

ENTRYPOINT [ "python", "train.py" ]
