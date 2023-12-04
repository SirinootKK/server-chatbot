FROM python:3.11.3

WORKDIR /app
COPY ./requirement.txt /app
RUN pip install -r requirement.txt

COPY . .
EXPOSE 5005
CMD ["gunicorn", "--bind", "0.0.0.0:5006", "app:run", "--timeout=1000"]