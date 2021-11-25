FROM tensorflow/tensorflow:2.7.0-gpu-jupyter
COPY requirements_dockerfile.txt /tmp/requirements.txt
RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y
RUN pip install --upgrade pip
RUN pip install -r /tmp/requirements.txt