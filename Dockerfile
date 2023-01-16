FROM python:3.11
LABEL author "Abhijith Ganesh"

RUN apt-get update -y && apt-get upgrade -y
RUN apt-get install postgresql postgresql-contrib -y

COPY "user-side-service" "service_one"
COPY "image-processing-service" "service_two"

RUN pip install -r service_one/requirements.txt

ENTRYPOINT [ "hypercorn", "service_one.app:app" , "--reload"]