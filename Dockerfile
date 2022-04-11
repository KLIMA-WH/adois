FROM tensorflow/tensorflow:2.5.0-gpu

RUN apt-get update
RUN apt-get install -y git

COPY requirements.txt .

RUN python -m pip install -r requirements.txt --ignore-installed --no-warn-script-location --upgrade