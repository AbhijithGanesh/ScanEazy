FROM debian:latest
LABEL author AbhijithGanesh

COPY . .
RUN apt-get update -y && apt-get upgrade -y
RUN apt-get install postgresql postgresql-contrib
