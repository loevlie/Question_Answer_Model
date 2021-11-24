FROM ubuntu:21.04

ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8

ENTRYPOINT ["/bin/bash", "-c"]
# RUN bash Anaconda-latest-Linux-x86_64.sh

RUN apt-get update && apt-get install -y curl
ARG DEBIAN_FRONTEND="noninteractive"
ENV TZ=US/Eastern
RUN apt-get update && apt-get install -y     python3     python3-pip

# RUN python3 -m pip --no-cache-dir install --upgrade     "pip<20.3"     setuptools
# RUN ln -s $(which python3) /usr/local/bin/python  
# RUN apt -y install curl 
# RUN curl https://repo.anaconda.com/archive/Anaconda3-2021.11-Linux-x86_64.sh

# RUN apt -y update
# RUN apt -y install software-properties-common
# RUN add-apt-repository ppa:deadsnakes/ppa
# RUN apt -y update
# RUN apt-get -y update
# RUN apt-get -y upgrade
# RUN apt -y install python3.9
# RUN apt -y install python3.9-dev
# RUN apt -y install python3.9-distutils


# RUN apt -y install python3-pip
# RUN apt -y install curl
# RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
# RUN python3.9 get-pip.py
# RUN echo "alias python3='python3.9'" >> /root/.bashrc
# RUN . ~/.bashrc
# RUN echo "alias pip3='pip3.9'" >> /root/.bashrc
# RUN . ~/.bashrc


# RUN apt -y install python3-distutils 
# RUN apt -y install curl 
# RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
# RUN python3 get-pip.py
# RUN echo "alias pip='pip3'" >> /root/.bashrc
# RUN . ~/.bashrc
# RUN pip3 install numpy
# RUN apt -y install python3.8-pip
# RUN pip3 install --upgrade pip

RUN apt-get update

RUN pip3 install spacy==3.1.0
# RUN python3 -m spacy download en_core_web_md
# RUN python3 -m spacy download en_core_web_sm

RUN pip3 install inflect

RUN pip3 install numpy

RUN pip3 install torch

RUN apt-get -y install git

RUN apt-get update
# RUN pip3 install tensorflow
# RUN pip3 install --upgrade tensorflow
# RUN pip3 install keras
# RUN mkdir ~/tf
# RUN cd ~/tf
# RUN python3 -m venv tensorflow-dev
# RUN source tensorflow-dev/bin/activate

RUN pip3 install git+https://github.com/msg-systems/coreferee.git#egg=coreferee
RUN pip3 install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.1.0/en_core_web_sm-3.1.0.tar.gz
RUN python3 -m coreferee install en

# RUN mkdir /QA

# ADD data /QA
# ADD test_questions.txt /QA
# ADD answer /QA
# ADD ask /QA
# ADD binaryAnswers.py /QA
# ADD dennyCode_modified.py /QA
# ADD helpers.py /QA
# ADD model.py /QA
# ADD preprocess.py /QA
# ADD QAfeatures.py /QA
# ADD Utils.py /QA

# WORKDIR /QA

# RUN ["chmod", "+x", "./answer"]
# RUN ["chmod", "+x", "./ask"]