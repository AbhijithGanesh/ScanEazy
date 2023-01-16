FROM postgres:latest
LABEL author AbhijithGanesh

COPY . .
RUN apt-get update -y && apt-get upgrade -y
RUN apt-get install python3 python3-pip -y

RUN python3 -m pip install -r requirements.txt

ENTRYPOINT [ "hypercorn", "app:app" ]
