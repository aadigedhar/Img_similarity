# Finding Visually Similar Garments for an Input Garment

In this find image similarity using [MobileNet](https://arxiv.org/abs/1704.04861) deep neural network.

Image similarity is a task mostly about feature selection of the image. Here, the Convolutional Neural Network (CNN) is used to extract features of these images. It is a better way for computer to understand them effectively.

This repository use a light-weight model, the MobileNet, to extract image features, then calculate their cosine distances as matrixes. The distance of two features will lie in `[-1, 1]`, where `-1` denotes the features are the most unlike, and `1` denotes they are the most similar. Choose a proper threshold `[-1, 1]`, the most similar images will be matched.

## Usage

The code is written to match the similar images in a huge amount as efficiently as possible.

```python
Run below commend in same dir of python folder

        python main.py -i " input img full-path" -d "Database img full-path"

Output result : In "Output_Img" folder
```

The requirements are also listed down bellow.

- tensorflow: the newest version for CPU, or the version that matches your GPU and CUDA.
- h5py
- numpy

## open-source GitHub repository used

https://github.com/ryanfwy/image-similarity

## Other options

We can also try different type of technolgy depend on Hardware, time, data, use. That are listed below :-

1. Resnet
2. siamese networks
