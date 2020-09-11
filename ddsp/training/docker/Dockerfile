# Uses one of AI Platform base images.
# You can try using different images however only this one has been tested.
FROM gcr.io/deeplearning-platform-release/tf2-gpu.2-2

# Installs sndfile library for reading and writing audio files.
RUN apt-get update && \
    apt-get install --no-install-recommends -y libsndfile-dev

# Upgrades Tensorflow and Tensorflow Probability
# Newer version of Tensorflow is needed for multiple VMs training
RUN pip install --upgrade pip && \
    pip install --upgrade tensorflow tensorflow-probability

# Installs cloudml-hypertune package needed for hyperparameter tuning
RUN pip install cloudml-hypertune

WORKDIR /root
# Installs Magenta DDSP from Github.
RUN wget https://github.com//magenta/ddsp/archive/master.zip && \
    unzip master.zip && \
    cd ddsp-master && \
    python setup.py install

# Copies running script.
COPY task.py task.py

ENTRYPOINT ["python", "task.py"]
