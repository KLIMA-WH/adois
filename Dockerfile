FROM tensorflow/tensorflow:2.5.0-gpu

RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub
RUN apt-get update
RUN apt-get install -y git

COPY requirements.txt .

RUN python -m pip install -r requirements.txt --ignore-installed --no-warn-script-location --upgrade