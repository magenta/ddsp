# Gin Configs: Papers

This directory contains gin configs for replicating the experiments in published
papers using ddsp. Each config file specifies both the dataset and the model,
so only a single --gin_file flag is required.


## List of papers

* [iclr2020](./iclr2020/): Experiments from the orginal DDSP ICLR 2020 paper ([paper](https://openreview.net/forum?id=B1x1ma4tDr), [blog](https://magenta.tensorflow.org/ddsp)).

```latex
@inproceedings{
  engel2020ddsp,
  title={DDSP: Differentiable Digital Signal Processing},
  author={Jesse Engel and Lamtharn (Hanoi) Hantrakul and Chenjie Gu and Adam Roberts},
  booktitle={International Conference on Learning Representations},
  year={2020},
  url={https://openreview.net/forum?id=B1x1ma4tDr}
}
```
