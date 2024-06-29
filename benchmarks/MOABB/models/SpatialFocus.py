import torch
from torch import nn


class SpatialFocus(nn.Module):
    def __init__(
        self,
        n_focal_points,
        focus_dims=3,
        similarity_func="cosine",
        similarity_transform=nn.Softmax(-1),
    ):
        super().__init__()
        self.n_focal_points = n_focal_points
        self.focus_dims = focus_dims
        self.similarity_func = similarity_func
        self.similarity_transform = similarity_transform

        self.focal_points = nn.Embedding(
            num_embeddings=n_focal_points, embedding_dim=self.focus_dims
        )

    def forward(self, x: torch.Tensor, positions: torch.Tensor):
        focal_points = self.focal_points.weight

        if self.similarity_func == "cosine":
            similarity = (positions @ focal_points.T) / (
                positions.norm(dim=-1).unsqueeze(1)
                @ focal_points.norm(dim=-1).unsqueeze(0)
            )
        else:
            similarity = self.similarity_func(positions, focal_points)

        if self.similarity_transform:
            weights = self.similarity_transform(similarity)
        else:
            weights = similarity

        weights = weights.unsqueeze(0)  # Batch dimension
        if x.dim() == 4:
            weights = weights.unsqueeze(1)  # Time dimension
        weights = weights.unsqueeze(-1)  # Feature dimension

        x = x.unsqueeze(-2)  # Focus dimension
        x = (x * weights).sum(dim=-2)

        return x


def inject_spatial_focus(
    model: nn.Module, spatial_focus: SpatialFocus, after: str
):
    seq, module_name = after.split(".")

    module: nn.Sequential = getattr(model, seq)
    new_module = nn.Sequential()
    for idx, (name, mod) in enumerate(module.named_modules()):
        new_module.append(mod)
        if name == module_name:
            new_module.append(mod)

    setattr(model, seq, new_module)
    return model
