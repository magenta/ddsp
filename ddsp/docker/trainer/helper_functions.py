"""Helper functions for interatcing with Magenta DDSP internals.
"""

import json
import os
from absl import logging
from ddsp.training import train_util
import tensorflow.compat.v2 as tf
from google.cloud import storage

def get_strategy(tpu='', gpus=None):
    """Chooses a distribution strategy.

    AI Platform automatically sets TF_CONFIG environment variable based 
    on provided config file. If training is run on multiple VMs different strategy
    needs to be chosen than when it is run on only one VM. This function determines 
    the strategy based on the information in TF_CONFIG variable.

    Args:
        tpu: 
            Argument for DDSP library function call.
            Address of the TPU. No TPU if left blank.
        gpus: 
            Argument for DDSP library function call.
            List of GPU addresses for synchronous training.
    Returns:
        A distribution strategy.
    """
    
    if 'TF_CONFIG' in os.environ:
        tf_config_str = os.environ.get('TF_CONFIG')
        logging.info("TFRecord %s", tf_config_str)
        tf_config_dict = json.loads(tf_config_str)

        # Exactly one chief worker is always specified inside the TF_CONFIG variable
        # in the cluster section. If there are any other workers specified MultiWorker
        # strategy needs to be chosen.
        if len(tf_config_dict["cluster"]) > 1:
            strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy()
            logging.info('Cluster spec: %s', strategy.cluster_resolver.cluster_spec())
        else:
            strategy = train_util.get_strategy(tpu=tpu, gpus=gpus)
    else:
        strategy = train_util.get_strategy(tpu=tpu, gpus=gpus)

    return strategy

def copy_config_file_from_gstorage(gstorage_path, container_path):
    """Downloads configuration path from the bucket to the container.

    Args:
        gstorage_path: 
            Path to the file inside the bucket that needs to be downloaded.
            Format: gs://bucket-name/path/to/file.txt
        container_path: 
            Path inside the container where downloaded file should be stored.
    """

    gstorage_path = gstorage_path.strip('gs:/')
    bucket_name = gstorage_path.split('/')[0]
    blob_name = os.path.relpath(gstorage_path, bucket_name)

    print(bucket_name, blob_name)

    storage_client = storage.Client()

    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_name)

    blob.download_to_filename(container_path)
    logging.info('Downloaded config file inside the container. Current location: %s', container_path)
