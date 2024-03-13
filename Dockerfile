FROM python:3.9-bookworm

WORKDIR /app

COPY ./requirements.txt /app

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

COPY . .

EXPOSE 5006

CMD ["waitress-serve", "--listen=*:5006", "run:flaskApp"]