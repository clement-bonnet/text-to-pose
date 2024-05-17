import abc
from typing import Any, Optional

import torch
from datasets import load_dataset
from tqdm.auto import tqdm
from transformers import AutoProcessor, CLIPModel

from clapp.model import CLaPPModel
from clapp.clapp_dataset import CLaPPDataset
from utils import draw_pose_pil_center_crop


class KNNClipTextToPose:
    def __init__(
        self,
        dataset: CLaPPDataset,
        device: torch.device | None = None,
        dtype: torch.dtype = torch.float32,
        clip_model: CLIPModel | None = None,
        clip_processor: AutoProcessor | None = None,
        distance: str = "l2",
        dataset_size_percentage: int = 100,
    ) -> None:
        self.device = device or torch.device("cpu")
        self.dataset = dataset
        self.all_text_features = torch.from_numpy(dataset.text_features).to(self.device)
        self.dtype = dtype
        self.clip_model = clip_model or CLIPModel.from_pretrained(
            "openai/clip-vit-large-patch14", torch_dtype=self.dtype
        )
        self.clip_model.to(self.device).eval()
        self.clip_processor = clip_processor or AutoProcessor.from_pretrained(
            "openai/clip-vit-large-patch14"
        )
        self.full_dataset = load_dataset(
            "clement-bonnet/test", split=f"train[:{dataset_size_percentage}%]"
        ).with_format("torch")

        self.distance = distance
        if self.distance == "l2":
            self.compute_distance = self._compute_distance_l2
        elif self.distance == "cosine":
            self.compute_distance = self._compute_distance_cosine
        else:
            raise ValueError

    def _compute_distance_l2(self, text_features: torch.Tensor) -> torch.Tensor:
        return torch.norm(self.all_text_features - text_features, dim=-1, p=2)

    def _compute_distance_cosine(self, text_features: torch.Tensor) -> torch.Tensor:
        return 1 - torch.cosine_similarity(self.all_text_features, text_features, dim=-1)

    def check_call_inputs(
        self,
        prompt: Optional[str | list[str]] = None,
        text_features: Optional[torch.Tensor] = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        if prompt is None:
            if text_features is None:
                raise ValueError("prompt (arg 0) and text_features (arg 1) cannot be both None.")
        else:
            if text_features is not None:
                raise ValueError(
                    "prompt (arg 0) and text_features (arg 1) cannot be provided "
                    "together, please set either to None."
                )

    @torch.no_grad()
    def __call__(
        self,
        prompt: Optional[str | list[str]] = None,
        clip_vit_l_14_text_features: Optional[torch.Tensor] = None,
        generator: Optional[torch.Generator] = None,
        num_poses_per_prompt: int = 1,
    ) -> tuple[dict, dict]:
        del generator
        self.check_call_inputs(prompt, clip_vit_l_14_text_features)
        if prompt is not None:
            assert isinstance(prompt, str) or len(prompt) == 1, "multiple prompts not supported yet"
            input_ids = self.clip_processor(
                text=prompt, return_tensors="pt", truncation=True, padding=True
            )["input_ids"]
            text_features = self.clip_model.get_text_features(input_ids.to(self.device)).squeeze(0)
        else:
            assert clip_vit_l_14_text_features.ndim == 1, "multiple prompts not supported yet"
            if clip_vit_l_14_text_features.shape[-1] != 768:
                raise ValueError(
                    "clip_vit_l_14_text_features (arg 1) does not have the right dimensions, expected last dimension "
                    f"to be 768, got tensor of shape {clip_vit_l_14_text_features.shape}."
                )
            text_features = clip_vit_l_14_text_features
        distances = self.compute_distance(text_features)
        _, indices = torch.topk(distances, num_poses_per_prompt, largest=False, sorted=True)
        keys = self.dataset.keys[indices.cpu().numpy()]
        try:
            query = self.full_dataset.filter(lambda x: x["key"] in keys)
        except Exception as ex:
            print(ex)
            print("key not found in dataset, keys:", keys)
            return None, None
        items = [query.filter(lambda x: x["key"] == key)[0] for key in keys]
        if len(items) != len(keys):
            raise ValueError(
                f"Some keys were not found in the full dataset. Got {len(items)} rows and keys: {keys}."
            )

        poses, img_poses, sizes, captions = [], [], [], []
        for item in items:
            pose = item["poses"]
            poses.append(pose)
            width, height = item["size"]
            img_pose = draw_pose_pil_center_crop(pose, height / width)
            img_poses.append(img_pose)
            sizes.append((width, height))
            caption = item["caption"]
            captions.append(caption)

        if num_poses_per_prompt == 1:
            poses, img_poses, sizes, captions, keys = (
                poses[0],
                img_poses[0],
                sizes[0],
                captions[0],
                keys[0],
            )
        info = dict(img_poses=img_poses, captions=captions, sizes=sizes, keys=keys)
        return poses, info


if __name__ == "__main__":
    dataset = CLaPPDataset("clapp/data", split="train", split_ratio=1.0)
    knn = KNNClipTextToPose(dataset)
    poses, info = knn("a person standing in the snow")
    print("poses:", poses)
    print("info:", info)
    info["img_poses"].show()
