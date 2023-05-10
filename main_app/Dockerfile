FROM python:3.8


WORKDIR /app

COPY requirements.txt .
RUN apt-get update
RUN apt-get -y install libusb-1.0-0-dev
RUN apt-get -y install libusb-1.0-0

RUN pip3 install -r requirements.txt
COPY . .

EXPOSE 5000
CMD ["gunicorn", "--workers", "10", "--max-requests", "10000", "-b", "0.0.0.0:5000", "--worker-connections", "1000", "--preload", "--log-level", "debug", "server:app"]