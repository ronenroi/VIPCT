from typing import Tuple
import torch
from .harmonic_embedding import HarmonicEmbedding





class FeatureEncoding(torch.nn.Module):
    def __init__(
        self,
        n_harmonic_functions: int = 64,
        **kwargs,
    ):
        """
        Args:
            n_harmonic_functions_xyz: The number of harmonic functions
                used to form the harmonic embedding of 3D point locations.
            n_harmonic_functions_dir: The number of harmonic functions
                used to form the harmonic embedding of the ray directions.
        """
        super().__init__()

        # The harmonic embedding layer converts input 3D coordinates
        # to a representation that is more suitable for
        # processing with a deep neural network.
        self.harmonic_embedding = HarmonicEmbedding(n_harmonic_functions, logspace=False)


    def forward(
        self,
        features: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """
        The forward function embeds the 3D points and rays normalized directions
        sampled along projection rays in camera coordinate system.


        Args:
            features: A tensor of shape `(minibatch, n_cam, n_feature)`.

        Returns:
            embeded_feature: A tensor of shape `(minibatch x, n_feature)`
                represents the 3D points embedding.
        """

        return torch.mean(self.harmonic_embedding(features),1)

