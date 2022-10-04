![image](https://user-images.githubusercontent.com/46085301/193828176-87d36dd3-b2a0-4d61-883d-ecfd856ac892.png)

# Transformers and ViT using PyTorch from scratch

This repository contains the implementation of transformers and Vision
Transformer using PyTorch from scratch with some examples.
Moreover, ViT is also trained on a custom dataset using pretrained weigths.

> This repo is **_still under development_**.

## Installation & Setup

```bash
  pip install git+https://github.com/bhimrazy/transformers-and-vit-using-pytorch-from-scratch
```

```bash
    #create a python environment
    $ python -m venv venv
    #activate environment
    $ source venv/bin/activate # use venv/Scripts/activate for windows
    #install packages from requirements.txt file
    $ pip install -r requirements.txt
```

## Usage/Examples

```shell
  # ViT Example
  $ python vit/predict.py -h
  # usage: predict.py [-h] [--path PATH]

  # ViT Imagenet Classifier

  # optional arguments:
  #   -h, --help            show this help message and exit
  #   --path PATH, -i PATH  image path

  $ python vit/predict.py --path cat.png
  # Predicting...

  # Classes Probabilities.
  # 0: tabby, tabby_cat                              --- 0.8001
  # 1: tiger_cat                                     --- 0.1752
  # 2: Egyptian_cat                                  --- 0.0172
  # 3: lynx, catamount                               --- 0.0018
  # 4: Persian_cat                                   --- 0.0011
```

## 📚References:

- Vaswani et al. (2017). Attention Is All You Need. arXiv. https://doi.org/10.48550/arXiv.1706.03762
- Dosovitskiy et al. An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale. arXiv. https://doi.org/10.48550/arXiv.2010.11929
- Wightman, R. (2019). PyTorch Image Models. GitHub repository. https://github.com/rwightman/pytorch-image-models
- mildlyoverfitted. (2021, March 5). Vision Transformer in PyTorch [Video]. YouTube. https://www.youtube.com/watch?v=ovB0ddFtzzA
- Yannic Kilcher. (2020, October 4). An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale [Video]. YouTube. https://www.youtube.com/watch?v=TrdevFK_am4

## Authors

- Bhimraj Yadav ([@bhimrazy](https://www.github.com/bhimrazy))
