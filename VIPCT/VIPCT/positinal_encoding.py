from typing import Tuple
import torch
from .harmonic_embedding import HarmonicEmbedding





class PositionalEncoding(torch.nn.Module):
    def __init__(
        self,
        n_harmonic_functions_xyz: int = 6,
        n_harmonic_functions_dir: int = 4,
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
        self.harmonic_embedding_xyz = HarmonicEmbedding(n_harmonic_functions_xyz)
        self.harmonic_embedding_dir = HarmonicEmbedding(n_harmonic_functions_dir)
        self.embedding_dim_xyz = n_harmonic_functions_xyz * 2 * 3 + 3
        self.embedding_dim_dir = n_harmonic_functions_dir * 2 * 3 + 3

    def forward(
        self,
        points: torch.Tensor,
        directions: torch.Tensor,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        The forward function embeds the 3D points and rays normalized directions
        sampled along projection rays in camera coordinate system.


        Args:
            points: A tensor of shape `(minibatch, ..., 3)` denoting the
                    3D points of the sampled rays in camera coords.
                directions: A tensor of shape `(minibatch, ..., 3)`
                    containing the normalized direction vectors of the sampled rays in camera coords.

        Returns:
            embeds_xyz: A tensor of shape `(minibatch x ... x self.n_harmonic_functions_dir*6 + 3)`
                represents the 3D points embedding.
            embeds_dir: A tensor of shape `(minibatch x ... x self.n_harmonic_functions_xyz*6 + 3)`
                represents the normalized directions embedding.
        """
        # We first convert the ray parametrizations to world
        # coordinates with `ray_bundle_to_ray_points`.
        # rays_points_world = ray_bundle_to_ray_points(ray_bundle)
        # rays_points_world.shape = [minibatch x ... x 3]
        directions = self.harmonic_embedding_dir(directions)
        # embeds_xyz.shape = [minibatch x ... x self.n_harmonic_functions_dir*6 + 3]
        # For each 3D world coordinate, we obtain its harmonic embedding.
        points = self.harmonic_embedding_xyz(points)
        # embeds_xyz.shape = [minibatch x ... x self.n_harmonic_functions_xyz*6 + 3]

        return points, directions

