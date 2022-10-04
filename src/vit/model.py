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


class Attention(nn.Module):
    """Attention mechanism.

    Parameters
    ----------
    dim : int
        The input and out dimension pf per token features.

    n_heads : int
        Number of attention heads

    qkv_bias : bool
        If True then we include bias to the query, key and value projections.

    attn_p : float
        Dropout probability applied to the query, key and value tensors.

    proj_p : float
        Dropout probability applied to the output tensor.

    Attributes
    ----------
    scale : float
        Normalizing constant for the dot product.

    qkv : nn.Linear
        Linear mapping that takes in the concatenated output of all attention
        heads and maps it into a new space.

    attn_drop, prod_drop : nn.Dropout
        Dropout layers.
    """

    def __init__(self, dim, n_heads, qkv_bias=True, attn_p=0., proj_p=0.):
        super(Attention, self).__init__()
        self.n_heads = n_heads
        self.dim = dim
        self.head_dim = dim // n_heads
        self.scale = self.scale = self.head_dim ** -0.5  # to prevent extreme large values to softmax

        self.qkv = nn.Linear(dim, dim*3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_p)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_p)

    def forward(self, x):
        """Run forward pass

        Parameters:
        -----------
            x : torch.Tensor
                Shape `(n_samples, n_patches + 1,dim)`.

        Returns
        -------
        torch.Tensor
            Shape`(n_samples, n_patches + 1,dim)`.
        """
        n_samples, n_tokens, dim = x.Shape

        if dim != self.dim:
            raise ValueError

        qkv = self.qkv(x)  # (n_samples, n_patches + 1,3 * dim)
        qkv = self.reshape(
            n_samples, n_tokens, 3, self.n_heads, self.head_dim
        )  # (n_samples, n_patches + 1, 3, n_heads, head_dim)
        qkv = qkv.permute(
            2, 0, 3, 1, 4
        )  # (3, n_samples, n_heads, n_patches+1, head_dim)
