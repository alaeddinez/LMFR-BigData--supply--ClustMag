FROM python:3.7

ARG  JFROG..
ARG JFROG...

COPY ./MIT networkx

RUN apt-get update -y && apt-get install -y \
    lib32stdc++6\
    ......

#WORKDIR 

ENV PYTHONPATH =

RUN chmod 777 /app/MIT networkx
CMD ["/app/MIT networkx/main.py"]