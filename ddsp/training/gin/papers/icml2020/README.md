# Self-supervised Pitch Detection by Inverse Audio Synthesis
_ICML SAS Workshop 2020_ ([paper](https://openreview.net/forum?id=RlVTYWhsky7))

Instructions for reproducing the results of the paper.

## Generate synthetic dataset

After pip installing ddsp, `pip install -U ddsp[data_preparation]`, you can create a synthetic dataset for the transcribing autoencoder using the installed script.
The command below creates a very small dataset as a test.

```bash
ddsp_generate_synthetic_dataset \
--output_tfrecord_path=/tmp/synthetic_data.tfrecord \
--gin_param="generate_examples.generate_fn=@data_preparation.synthetic_data.generate_notes_v2" \
--gin_param="generate_notes_v2.n_timesteps=125" \
--gin_param="generate_notes_v2.n_harmonics=100" \
--gin_param="generate_notes_v2.n_mags=65" \
--num_examples=1000 \
--num_shards=2 \
--alsologtostderr
```

The synthetic dataset in the paper has 10,000,000 examples split into 1000 shards.
Because that's a large dataset, we have provided sharded TFRecord files for the synthetic on GCP at `gs://ddsp-inv/datasets/notes_t125_h100_m65_v2.tfrecord*`.
As the name indicates, the dataset was made with 125 timesteps, 100 harmonics, and 65 noise bins.
If training on GCP it is fast to directly read from these buckets, but if training locally you will probably want to download the files locally (~1.7 TB) using the `gsutil` command line utility from the [gcloud sdk](https://cloud.google.com/sdk/docs/downloads-interactive).


If you wish to make your own large dataset, it may require runinng in a distributed setup such as [Dataflow](https://cloud.google.com/dataflow) on GCP.
If running on DataFlow, you'll need to set the `--pipeline_options` flag using the execution parameters described at https://cloud.google.com/dataflow/docs/guides/specifying-exec-params
E.g., `--pipeline_options=--runner=DataFlowRunner,--project=<my-gcp-project-id>,--temp_location=gs://<bucket/directory>,--region=us-central1,--job_name=<my-name-of-job>`...

## Pretrain transcribing autoencoder

### Train
Point the dataset file_pattern to the synthetic dataset created above.

```bash
ddsp_run \
--mode=train \
--save_dir=/tmp/$USER-tae-pretrain-0 \
--gin_file=papers/icml2020/pretrain_model.gin \
--gin_file=papers/icml2020/pretrain_dataset.gin \
--gin_param="SyntheticNotes.file_pattern='gs://ddsp-inv/datasets/notes_t125_h100_m65_v2.tfrecord*'" \
--gin_param="batch_size=64" \
--alsologtostderr
```

This command points to datasets on GCP, and the gin_params for file_pattern are redudant with the default values in the gin files, but provided here to show how you would modify them for local dataset paths.

In the paper we train for ~1.2M steps with a batch size of 64. A single v100 can fit a max batch size of 32, so you will need to use multiple gpus to exactly reproduce the experiment. Given the large amount of pretraining, a pretrained checkpoint [is available here](https://storage.googleapis.com/ddsp-inv/ckpts/synthetic_pretrained_ckpt.zip)
or on GCP at `gs://ddsp-inv/ckpts/synthetic_pretrained_ckpt`.

### Eval and Sample

This will add evaluation metrics as scalar summaries for tensorboard.

```bash
ddsp_run \
--mode=eval \
--run_once \
--restore_dir=/path/to/trained_ckpt \
--save_dir=/path/to/trained_ckpt \
--gin_file=papers/icml2020/pretrain_dataset.gin \
--alsologtostderr
```

This will add image and audio scalar summaries for tensorboard.

```bash
ddsp_run \
--mode=sample \
--run_once \
--restore_dir=/path/to/trained_ckpt \
--save_dir=/path/to/trained_ckpt \
--gin_file=papers/icml2020/pretrain_dataset.gin \
--alsologtostderr
```


## Finetuning transcribing autoencoder

### Train
Now we finetune the model from above on a specific dataset. Use the `--restore_dir` flag to point to your pretrained checkpoint.

A pretrained model on 1.2M steps (batch size=64) of synthetic data [is available here](https://storage.googleapis.com/ddsp-inv/ckpts/synthetic_pretrained_ckpt.zip)
or on GCP.

```bash
gsutil cp -r gs://ddsp-inv/ckpts/synthetic_pretrained_ckpt /path/to/synthetic_pretrained_ckpt
```

```bash
ddsp_run \
--mode=train \
--restore_dir=/path/to/synthetic_pretrained_ckpt \
--save_dir=/tmp/$USER-tae-finetune-0 \
--gin_file=papers/icml2020/finetune_model.gin \
--gin_file=papers/icml2020/finetune_dataset.gin \
--gin_param="SyntheticNotes.file_pattern='gs://ddsp-inv/datasets/notes_t125_h100_m65_v2.tfrecord*'" \
--gin_param="train_data/TFRecordProvider.file_pattern='gs://ddsp-inv/datasets/all_instruments_train.tfrecord*'" \
--gin_param="test_data/TFRecordProvider.file_pattern='gs://ddsp-inv/datasets/all_instruments_test.tfrecord*'" \
--gin_param="batch_size=64" \
--alsologtostderr
```

This command points to datasets on GCP, and the gin_params for file_pattern are redudant with the default values in the gin files, but provided here to show how you would modify them for local dataset paths.
We have provided sharded TFRecord files for the [URMP dataset](http://www2.ece.rochester.edu/projects/air/projects/URMP/annotations_5P.html) on GCP at `gs://ddsp-inv/datasets/all_instruments_train.tfrecord*` and `gs://ddsp-inv/datasets/all_instruments_test.tfrecord*`.
If training on GCP it is fast to directly read from these buckets, but if training locally you will probably want to download the files locally (~16 GB) using the `gsutil` command line utility from the [gcloud sdk](https://cloud.google.com/sdk/docs/downloads-interactive).


In the paper, this model was trained with a batch size of 64 on 8 accelerators (8 per an accelerator), and typically converges after 200-400k iterations. A single v100 can fit a max batch size of 12, so you will need to use multiple GPUs or TPUs to exactly reproduce the experiment. To use a TPU, start up an instance from the web interface and pass the internal ip address to the tpu flag `--tpu=grpc://<internal-ip-address>`.


Finetuned models for +400k steps (batch size=64) are available on GCP the
[URMP](http://www2.ece.rochester.edu/projects/air/projects/URMP/annotations_5P.html) ([ckpt](https://storage.googleapis.com/ddsp-inv/ckpts/urmp_ckpt.zip)),
[MDB-stem-synth](https://zenodo.org/record/1481172#.Xzouy5NKhTY) ([ckpt](https://storage.googleapis.com/ddsp-inv/ckpts/mdb_stem_synth_ckpt.zip)),
and [MIR1k](https://sites.google.com/site/unvoicedsoundseparation/mir-1k) ([ckpt](https://storage.googleapis.com/ddsp-inv/ckpts/mir1k_ckpt.zip)) datasets.

### Eval and Sample

This will add evaluation metrics as scalar summaries for tensorboard.

```bash
ddsp_run \
--mode=eval \
--run_once \
--restore_dir=/path/to/trained_ckpt \
--save_dir=/path/to/trained_ckpt \
--gin_file=papers/icml2020/finetune_dataset.gin \
--alsologtostderr
```

This will add image and audio scalar summaries for tensorboard.

```bash
ddsp_run \
--mode=sample \
--run_once \
--restore_dir=/path/to/trained_ckpt \
--save_dir=/path/to/trained_ckpt \
--gin_file=papers/icml2020/finetune_dataset.gin \
--alsologtostderr
```
