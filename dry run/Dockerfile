FROM ubuntu:18.04

ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8

RUN apt-get -y update && \
    apt-get -y upgrade && \
    apt-get -y install python-pip python3-dev

RUN pip3 install --upgrade pip
RUN pip3 install spacy==3.1.0
RUN python3 -m spacy download en_core_web_md
RUN python3 -m spacy download en_core_web_sm

RUN pip3 install inflect

RUN pip3 instalal numpy

RUN pip3 install torch

RUN pip3 install git+https://github.com/msg-systems/coreferee.git#egg=coreferee
RUN pip3 install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.1.0/en_core_web_sm-3.1.0.tar.gz
RUN python3 -m coreferee install en

RUN mkdir /QA

ADD Data /QA
ADD answer /QA
ADD ask /QA
ADD binaryAnswers.py /QA
ADD dennyCode_modified.py /QA
ADD helpers.py /QA
ADD model.py /QA
ADD preprocess.py /QA
ADD QAfeatures.py /QA
ADD Utils.py /QA

ADD Question_Gen_Model /QA

CMD ["chmod a+x /QA*"]

WORKDIR /QA
ENTRYPOINT ["/bin/bash", "-c"]
