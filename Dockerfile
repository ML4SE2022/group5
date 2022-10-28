FROM python:3.10

COPY requirements.txt .

RUN pip install --upgrade pip

RUN pip install -r requirements.txt

COPY 50k_types/ 50k_types/
COPY type-model/ type-model/
COPY src/ .

ENTRYPOINT [ "python", "train.py" ]
