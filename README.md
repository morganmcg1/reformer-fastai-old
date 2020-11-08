# Reformer Reproducibility Experiments

Fastai community entry to [2020 Reproducibility Challenge](https://paperswithcode.com/rc2020)

## Links
- [Fastai forums thread](https://forums.fast.ai/t/reproducibility-challenge-2020-fastai-folks-interested/80336/39)
- [Google doc](https://docs.google.com/document/d/1wF83E3B3yXIGZixEgOUJI2T2XXhT1DVCrPXS5Dbsyh8/edit)

## Resources

### Author's Resources
- [Reformer Paper](https://openreview.net/pdf?id=rkgNKkHtvB)
- [Authors ICLR video](https://iclr.cc/virtual_2020/poster_rkgNKkHtvB.html)
- [Google Blog](https://ai.googleblog.com/2020/01/reformer-efficient-transformer.html)
- [Authors code (TRAX)](https://github.com/google/trax/tree/master/trax/models/reformer)
- [Reformer enwik8 model and training config](https://github.com/google/trax/blob/f8024e8057599b92fce82842f342cb3d39c8f405/trax/supervised/configs/reformer_enwik8.gin)

### Code
- [@lucidrain’s Reformer code](https://github.com/lucidrains/reformer-pytorch/)
- [HuggingFace: Reformer source code](https://github.com/huggingface/transformers/blob/a1bbcf3f6c20e15fe799a8659d6b7bd36fdf11ed/src/transformers/modeling_reformer.py)
- [HuggingFace: Reformer notebook example](https://colab.research.google.com/github/patrickvonplaten/blog/blob/master/notebooks/03_reformer.ipynb)
- [HuggingFace: long sequences](https://colab.research.google.com/github/patrickvonplaten/notebooks/blob/master/PyTorch_Reformer.ipynb)
- [HuggingFace: Pretraining](https://colab.research.google.com/drive/1tzzh0i8PgDQGV3SMFUGxM7_gGae3K-uW?usp=sharing)

### Data
**enwik8**
- [enwik8.zip, raw data, 100mb](http://mattmahoney.net/dc/enwik8.zip)
- [Tensor2Tensor enwik8 data generator code, with train/dev/test split](https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/data_generators/enwik8.py). File lengths:
    - Train: 89,621,832
    - Eval: 5,000,000
    - Test: 5,000,000
- [enwik8 notebook Tensor2Tensor](https://github.com/morganmcg1/reformer-fastai/blob/main/enwiki8_Tensor2Tensor_download.ipynb)

### Explainers
- [Yannic K explainer](https://www.youtube.com/watch?v=i4H0kjxrias&t=1s)
- [HuggingFace blog post](https://huggingface.co/blog/reformer)
- [Illustrating the Reformer blog post](https://towardsdatascience.com/illustrating-the-reformer-393575ac6ba0)

### Related
- [Coursera Attention Models in NLP course, with Reformer co-author](https://www.coursera.org/learn/attention-models-in-nlp)
- [@hallvagi Attention blogpost](https://hallvagi.github.io/dl-explorer/fastai/attention/lstm/2020/06/29/Attention.html)

