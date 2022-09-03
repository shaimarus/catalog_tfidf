FROM python:3.8.8
RUN apt-get update
#RUN apt-get update && rm -rf /var/lib/apt/listls/*
RUN mkdir /opt/catalog_tfidf/
COPY . /opt/catalog_tfidf/

RUN pip3 install -r /opt/catalog_tfidf/requirements.txt
#RUN pip3 --no-cache-dir install -r /opt/catalog_tfidf/requirements.txt

WORKDIR /opt/catalog_tfidf/