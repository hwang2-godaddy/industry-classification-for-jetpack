# from GDDY's golden container
FROM 764525110978.dkr.ecr.us-west-2.amazonaws.com/debian:11-slim

USER root

RUN apt-get update && apt-get upgrade -y
RUN apt-get install -y --no-install-recommends python3-pip

# install python
RUN apt-get update && apt-get install -y python3-pip git curl
RUN ln -s /usr/bin/python3 /usr/bin/python

# install pytorch
RUN pip3 install torch torchvision torchaudio

# install transformers and datasets
RUN pip3 install transformers datasets

# install peft
RUN pip3 install peft

# install other useful packages
RUN pip3 install numpy pandas scikit-learn requests scipy

# install sagemaker-training toolkit
RUN apt-get install -y python3-dev
RUN apt-get install -y build-essential
RUN pip3 install sagemaker-training

# install ray AIR
RUN pip3 install ray[air]>=1.13

# copy the training script inside the container
# the training script must be located in the /opt/ml/code directory
COPY train.py /opt/ml/code/train.py

# NOTE: rm private key files to get container scan to pass
RUN rm -rf /usr/lib/**/site-packages/**/tests/

# set up working directory
WORKDIR /opt/ml/code

# define train.py as the script entry point
# Python and shell scripts are both supported
ENV SAGEMAKER_PROGRAM train.py
