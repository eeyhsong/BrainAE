"""BrainAE v0.2: MUA. Original model layers; portable release entry point."""

import torch
from torch import Tensor, nn
from torch.nn import init
from einops.layers.torch import Rearrange


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("Linear") != -1:
        init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


class PatchEmbedding(nn.Module):

    def __init__(self, emb_size=40):
        super().__init__()
        self.shallownet = nn.Sequential(
            nn.Conv2d(1, 40, (1, 26), (1, 1)),
            nn.AvgPool2d((1, 5), (1, 5)),
            nn.BatchNorm2d(40),
            nn.ELU(),
            nn.Conv2d(40, 40, (1024, 1), (1, 1)),
            nn.BatchNorm2d(40),
            nn.ELU(),
            nn.Dropout(0.5),
        )
        self.projection = nn.Sequential(
            nn.Conv2d(40, emb_size, (1, 1), stride=(1, 1)),
            Rearrange("b e (h) (w) -> b (h w) e"),
        )

    def forward(self, x: Tensor) -> Tensor:
        b, _, _, _ = x.shape
        x = self.shallownet(x)
        x = self.projection(x)
        x = x.contiguous().view(x.size(0), -1)
        return x


class ResidualAdd(nn.Module):

    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        x += res
        return x


class Proj_eeg(nn.Sequential):

    def __init__(self, embedding_dim=1800, proj_dim=1024, drop_proj=0.5):
        super().__init__(
            nn.Linear(embedding_dim, proj_dim),
            ResidualAdd(
                nn.Sequential(
                    nn.GELU(), nn.Linear(proj_dim, proj_dim), nn.Dropout(drop_proj)
                )
            ),
            nn.LayerNorm(proj_dim),
        )


class Enc_eeg(nn.Module):

    def __init__(
        self, input_shape=(1, 1024, 200), emb_size=40, emb_dim=1400, proj_dim=1024
    ):
        super().__init__()
        self.patch_embed = PatchEmbedding(emb_size=emb_size)
        self.proj_eeg = Proj_eeg(embedding_dim=emb_dim, proj_dim=proj_dim)

    def forward(self, x):
        x_embedded = self.patch_embed(x)
        x_projected = self.proj_eeg(x_embedded)
        return x_projected


class Dec_eeg(nn.Module):

    def __init__(
        self, input_shape=(1, 1024, 200), emb_size=40, emb_dim=1400, proj_dim=1024
    ):
        super().__init__()
        self.dec_eeg = nn.Sequential(
            nn.Linear(proj_dim, emb_dim),
            nn.GELU(),
            nn.Unflatten(1, (emb_size, 1, -1)),
            nn.ConvTranspose2d(40, 40, (1024, 1), stride=(1, 1)),
            nn.BatchNorm2d(40),
            nn.ELU(),
            nn.Upsample(scale_factor=(1, 5), mode="nearest"),
            nn.ConvTranspose2d(emb_size, 40, (1, 26), stride=(1, 1)),
            nn.BatchNorm2d(40),
            nn.ELU(),
            nn.ConvTranspose2d(40, 1, (1, 1), stride=(1, 1)),
            nn.BatchNorm2d(1),
            nn.Tanh(),
        )

    def forward(self, x):
        x_reconstructed = self.dec_eeg(x)
        return x_reconstructed


def main():
    from brainae.runner import run

    run("mua", Enc_eeg, Dec_eeg, weights_init_normal)


if __name__ == "__main__":
    main()
