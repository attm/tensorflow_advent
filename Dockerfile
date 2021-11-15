FROM tensorflow/tensorflow:2.7.0-gpu-jupyter
COPY requirements.txt /tmp/requirements.txt
RUN pip install --upgrade pip
RUN pip install -r /tmp/requirements.txt