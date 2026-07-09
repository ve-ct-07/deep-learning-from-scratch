# Datasets

Datasets are not stored in this repository. Download them separately and
place them in this directory.

## MNIST

Download the IDX files from [Yann LeCun's MNIST page](http://yann.lecun.com/exdb/mnist/)
and place them in `data/mnist_data/`:

```text
data/mnist_data/
├── train-images-idx3-ubyte(.gz)
├── train-labels-idx1-ubyte(.gz)
├── t10k-images-idx3-ubyte(.gz)
└── t10k-labels-idx1-ubyte(.gz)
```

## IMDB Sentiment Dataset

Download the [Large Movie Review Dataset](https://ai.stanford.edu/~amaas/data/sentiment/)
and extract it so it looks like:

```text
data/aclImdb/
├── train/
│   ├── pos/
│   └── neg/
└── test/
    ├── pos/
    └── neg/
```