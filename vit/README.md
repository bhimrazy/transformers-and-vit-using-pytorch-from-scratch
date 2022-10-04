# Vision Transformer
Introduced by Dosovitskiy et al. in An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale

The Vision Transformer, or ViT, is a model for image classification that employs a Transformer-like architecture over patches of the image. An image is split into fixed-size patches, each of them are then linearly embedded, position embeddings are added, and the resulting sequence of vectors is fed to a standard Transformer encoder. In order to perform classification, the standard approach of adding an extra learnable “classification token” to the sequence is used.

<p align="center">
  <img alt="Vision Transformer" src="https://user-images.githubusercontent.com/46085301/193829546-b7c202fd-34fe-424b-8c9f-37cf6e9675e2.png"/>
</P>


## 📚References:
- Vaswani et al. (2017). Attention Is All You Need. arXiv. https://doi.org/10.48550/arXiv.1706.03762
- Dosovitskiy et al. An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale. arXiv. https://doi.org/10.48550/arXiv.2010.11929
- Papers with Code - Vision Transformer Explained. (n.d.). https://paperswithcode.com/method/vision-transformer
- Wightman, R. (2019). PyTorch Image Models. GitHub repository. https://github.com/rwightman/pytorch-image-models
- mildlyoverfitted. (2021, March 5). Vision Transformer in PyTorch [Video]. YouTube. https://www.youtube.com/watch?v=ovB0ddFtzzA
- Yannic Kilcher. (2020, October 4). An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale [Video]. YouTube. https://www.youtube.com/watch?v=TrdevFK_am4
