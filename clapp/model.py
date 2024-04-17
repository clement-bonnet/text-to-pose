from typing import Optional, Union, Dict

from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from transformers import AutoProcessor, CLIPModel
from huggingface_hub import PyTorchModelHubMixin, hf_hub_download


class ResidualProjectionHead(nn.Module, PyTorchModelHubMixin):
    def __init__(self, embedding_dim: int, projection_dim: int, dropout: float):
        super().__init__()
        self.projection = nn.Linear(embedding_dim, projection_dim)
        self.dense = nn.Linear(projection_dim, projection_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        projected = self.projection(x)
        x = F.gelu(projected)
        x = self.dense(x)
        x = self.dropout(x)
        x = x + projected
        x = x / x.norm(dim=-1, keepdim=True)
        return x


class CLaPPModel(nn.Module, PyTorchModelHubMixin):
    def __init__(
        self,
        device: torch.device = torch.device("cpu"),
        target_temperature: float = 0.1,
        pose_embedding_dim: int = 768,
        text_embedding_dim: int = 768,
        projection_dim: int = 768,
        dropout: float = 0.1,
        scale_logit_temperature: float = 10.0,
    ):
        super().__init__()
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
        for p in self.clip_model.parameters():
            p.requires_grad = False
        self.clip_model.eval()
        self.processor = AutoProcessor.from_pretrained("openai/clip-vit-large-patch14")

        self.pose_projection = ResidualProjectionHead(pose_embedding_dim, projection_dim, dropout)
        self.text_projection = ResidualProjectionHead(text_embedding_dim, projection_dim, dropout)
        self.target_temperature = target_temperature
        self.scale_logit_temperature = scale_logit_temperature
        self.log_logit_temperature = torch.nn.Parameter(
            torch.tensor(np.log(1.0) / self.scale_logit_temperature, dtype=torch.float32)
        )
        self.device = device

    def eval(self):
        self.pose_projection.eval()
        self.text_projection.eval()
        return super().eval()

    def train(self, mode: bool = True):
        self.pose_projection.train(mode)
        self.text_projection.train(mode)
        return super().train(mode)

    def _save_pretrained(self, save_directory: Path) -> None:
        """Overwrite this method to save both text and pose projection modules."""
        # Save the pose projection layer
        pose_save_directory = save_directory / "pose_projection"
        pose_save_directory.mkdir(parents=True, exist_ok=True)
        self.pose_projection._save_pretrained(pose_save_directory)
        # Save the text projection layer
        text_save_directory = save_directory / "text_projection"
        text_save_directory.mkdir(parents=True, exist_ok=True)
        self.text_projection._save_pretrained(text_save_directory)

    @classmethod
    def _from_pretrained(
        cls,
        *,
        model_id: str,
        revision: Optional[str],
        cache_dir: Optional[Union[str, Path]],
        force_download: bool,
        proxies: Optional[Dict],
        resume_download: bool,
        local_files_only: bool,
        token: Union[str, bool, None],
        map_location: str = "cpu",
        strict: bool = False,
        **model_kwargs,
    ):
        model = cls(**model_kwargs)
        pose_projection_path = hf_hub_download(
            repo_id=model_id, filename="pose_projection/model.safetensors"
        )
        cls._load_as_safetensor(
            model.pose_projection,
            pose_projection_path,
            map_location=map_location,
            strict=strict,
        )
        text_projection_path = hf_hub_download(
            repo_id=model_id, filename="text_projection/model.safetensors"
        )
        cls._load_as_safetensor(
            model.text_projection,
            text_projection_path,
            map_location=map_location,
            strict=strict,
        )
        return model

    def get_embeddings(self, batch):
        pose_features = batch["pose_features"].to(self.device)
        text_features = batch["text_features"].to(self.device)
        pose_embeddings = self.pose_projection(pose_features)
        text_embeddings = self.text_projection(text_features)
        return pose_embeddings, text_embeddings

    def get_logits(self, pose_embeddings, text_embeddings):
        logit_temperature = torch.exp(self.scale_logit_temperature * self.log_logit_temperature)
        logits = (text_embeddings @ pose_embeddings.T) / logit_temperature
        return logits

    @torch.no_grad()
    def get_target(self, pose_embeddings, text_embeddings):
        poses_similarity = pose_embeddings @ pose_embeddings.T
        texts_similarity = text_embeddings @ text_embeddings.T
        targets = F.softmax(
            (poses_similarity + texts_similarity) / (2 * self.target_temperature),
            dim=-1,
        )
        return targets

    def forward(self, batch):
        # Getting the embeddings
        pose_embeddings, text_embeddings = self.get_embeddings(batch)

        # Calculating the Loss
        logits = self.get_logits(pose_embeddings, text_embeddings)
        targets = self.get_target(pose_embeddings, text_embeddings)
        texts_loss = cross_entropy(logits, targets, reduction="none")
        poses_loss = cross_entropy(logits.T, targets.T, reduction="none")
        loss = torch.mean((poses_loss + texts_loss) / 2.0)
        return loss

    def diagonal_precision(self, batch):
        pose_embeddings, text_embeddings = self.get_embeddings(batch)
        logits = self.get_logits(pose_embeddings, text_embeddings)
        arange = torch.arange(len(logits))
        text_is_correct = torch.isclose(
            logits[arange, arange], torch.max(logits, dim=-1).values, rtol=2e-2
        )
        pose_is_correct = torch.isclose(
            logits.T[arange, arange], torch.max(logits.T, dim=-1).values, rtol=2e-2
        )
        return torch.mean(pose_is_correct.float()), torch.mean(text_is_correct.float())

    def score(self, img_poses: Image.Image | list[Image.Image], text: str | list[str]):
        pose_inputs = self.processor(images=img_poses, return_tensors="pt")["pixel_values"]
        pose_features = self.clip_model.get_image_features(pose_inputs.to(self.device))
        input_ids = self.processor(text=text, return_tensors="pt", truncation=True)["input_ids"]
        text_features = self.clip_model.get_text_features(input_ids.to(self.device))
        pose_embeddings = self.pose_projection.to(self.device)(pose_features)
        text_embeddings = self.text_projection.to(self.device)(text_features)
        logits = text_embeddings @ pose_embeddings.T
        return logits


def cross_entropy(logits, targets, reduction="none") -> torch.Tensor:  # type: ignore
    loss = (-targets * F.log_softmax(logits, dim=-1)).sum(1)
    if reduction == "none":
        return loss
    elif reduction == "mean":
        return loss.mean()


if __name__ == "__main__":
    model = CLaPPModel.from_pretrained("clement-bonnet/clapp-v0")
    print(model)
