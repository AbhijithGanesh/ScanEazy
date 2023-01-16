FROM debian:latest

COPY . .

RUN apt-get update -y
RUN apt-get install ffmpeg libsm6 libxext6 python3 python3-pip -y
RUN python3 -m pip install -r requirements.txt

ENTRYPOINT [ "hypercorn", "app:app"]
EXPOSE 8000