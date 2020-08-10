FROM gcr.io/deeplearning-platform-release/tf2-gpu.2-2
WORKDIR /root

# Installs sndfile library for reading and writing audio files 
RUN apt-get update && apt-get install -y libsndfile-dev

# Installs Magenta DDSP library and upgrades Tensorflow
# Newer version of Tensorflow is needed for multiple VMs training
RUN pip install --upgrade pip && pip install --upgrade tensorflow ddsp
RUN pip show tensorflow

# Installs Google Cloud SDK, this is mostly for using gsutil to copy the configs.
RUN wget -nv \
    https://dl.google.com/dl/cloudsdk/release/google-cloud-sdk.tar.gz && \
    mkdir /root/tools && \
    tar xvzf google-cloud-sdk.tar.gz -C /root/tools && \
    rm google-cloud-sdk.tar.gz && \
    /root/tools/google-cloud-sdk/install.sh --usage-reporting=false \
        --path-update=false --bash-completion=false \
        --disable-installation-options && \
    rm -rf /root/.config/* && \
    ln -s /root/.config /config && \
    # Remove the backup directory that gcloud creates
    rm -rf /root/tools/google-cloud-sdk/.install/.backup

# Path configuration
ENV PATH $PATH:/root/tools/google-cloud-sdk/bin
# Makes sure gsutil will use the default service account
RUN echo '[GoogleCompute]\nservice_account = default' > /etc/boto.cfg

# Copies additional source code
COPY docker/trainer/ddsp_run.py /root/trainer/ddsp_run.py
COPY docker/trainer/helper_functions.py /root/trainer/helper_functions.py
COPY training/train_util.py /root/trainer/magenta_ddsp_internals/train_util.py

# Copies default gin configuration files
COPY docker/trainer/gin/optimization/base.gin /root/trainer/gin/optimization/base.gin
COPY docker/trainer/gin/models/solo_instrument.gin /root/trainer/gin/models/solo_instrument.gin 
COPY docker/trainer/gin/models/ae.gin /root/trainer/gin/models/ae.gin 
COPY docker/trainer/gin/datasets/tfrecord.gin /root/trainer/gin/datasets/tfrecord.gin
COPY docker/trainer/gin/datasets/base.gin /root/trainer/gin/datasets/base.gin

# These parameters can be also specified as part of the job submission arguments
ENTRYPOINT ["python", "trainer/ddsp_run.py", "--mode=train", \
     "--alsologtostderr", \
     "--gin_file=models/solo_instrument.gin", \
     "--gin_file=datasets/tfrecord.gin"]



