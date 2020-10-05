FROM python:3

WORKDIR ./
CMD brew install libomp
COPY requirements.txt ./
COPY ./Data ./
COPY ./3.SimpleLSTM.h5 ./
RUN pip install --no-cache-dir -r requirements.txt
COPY . .

CMD [ "python", "./test3.py" ]
