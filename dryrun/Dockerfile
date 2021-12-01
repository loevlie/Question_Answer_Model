FROM ubuntu:21.04

ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8

ENTRYPOINT ["/bin/bash", "-c"]

RUN apt-get update && apt-get install -y curl
ARG DEBIAN_FRONTEND="noninteractive"
ENV TZ=US/Eastern
RUN apt-get update && apt-get install -y     python3     python3-pip

RUN apt-get update

RUN pip3 install spacy==3.1.0
RUN python3 -m spacy download en_core_web_md
RUN python3 -m spacy download en_core_web_sm

RUN pip3 install inflect

RUN pip3 install numpy

RUN pip3 install torch

RUN apt-get -y install git

RUN apt-get update

RUN pip3 install git+https://github.com/msg-systems/coreferee.git#egg=coreferee
RUN pip3 install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.1.0/en_core_web_sm-3.1.0.tar.gz
RUN python3 -m coreferee install en

RUN mkdir /QA

ADD data /QA
ADD test_questions.txt /QA
ADD answer /QA
ADD ask /QA
ADD binaryAnswers.py /QA
ADD dennyCode_modified.py /QA
ADD helpers.py /QA
ADD model.py /QA
ADD preprocess.py /QA
ADD QAfeatures.py /QA
ADD QSanalysis.py /QA
ADD Utils.py /QA

WORKDIR /QA

RUN ["chmod", "+x", "./answer"]
RUN ["chmod", "+x", "./ask"]