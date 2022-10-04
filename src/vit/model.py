import torch
import torch.nn as nn


class PatchEmbed(nn.Module):
    """Split image into patches and then embed them.

    Parameters
    ----------
    img_size : int
        Size of the image (it is square).

    patch_size : int
        Size of the patch (it is a square).

    in_channels : int
        Number of input channels

    embedding_dim : int
        The embedding dimension.

    Attributes
    ----------
    n_patches : int 
        Number of patches inside of our image.

    proj : nn.Conv2d
        Convolutional layer (projection) that does both the splitting into patches
        and their embedding.
    """

    def __init__(self, img_size, patch_size, in_channels=3, embed_dim=768):
        super(PatchEmbed, self).__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2

        self.proj = nn.Conv2d(
            in_channels=in_channels,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

    def forward(self, x):
        """Run forward pass

        Parameters:
        -----------
            x : torch.Tensor
                Shape `(n_samples, in_channels, img_size, img_size)`.

        Returns
        -------
        torch.Tensor
            Shape`(n_samples, n_patches, embed_dim)`.
        """
        x = self.proj(
            x
        )  # (n_samples, embed_dim, n_patches ** 0.5, n_patches ** 0.5)
        x = x.flatten(2)  # (n_samples, embed_dim, n_patches)
        x = x.transpose(1, 2)  # (n_samples, n_patches, embed_dim)

        return x

