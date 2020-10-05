FROM python:3

WORKDIR ./
CMD brew install libomp
COPY requirements.txt ./
COPY ./Data ./
RUN pip install --no-cache-dir -r requirements.txt
COPY . .

CMD [ "python", "./test1.py" ]
