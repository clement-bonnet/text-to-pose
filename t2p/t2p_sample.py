from dataclasses import dataclass
from typing import Optional, Union

import numpy as np
import torch


def _to_device(
    x: Union[torch.Tensor, np.ndarray],
    device: torch.device,
    dtype: Optional[torch.dtype] = None,
):
    return x.to(device, dtype) if isinstance(x, torch.Tensor) else x


def _to_numpy(
    x: Union[torch.Tensor, np.ndarray],
):
    return x.cpu().numpy() if isinstance(x, torch.Tensor) else x


def _unsqueeze(x: Union[torch.Tensor, np.ndarray], dim: int):
    return x.unsqueeze(dim) if isinstance(x, torch.Tensor) else np.expand_dims(x, dim)


@dataclass(frozen=True)
class T2PSample:
    bboxes: Union[torch.Tensor, np.ndarray]  # (n, 2, 2)
    bodies: Union[torch.Tensor, np.ndarray]  # (n, 18, 2)
    faces: Union[torch.Tensor, np.ndarray]  # (n, 68, 2)
    hands: Union[torch.Tensor, np.ndarray]  # (n, 42, 2)
    pose_mask: Union[torch.Tensor, np.ndarray]  # (n,)
    image_ratio: Union[torch.Tensor, np.ndarray]  # () -> (width/height)
    input_ids: Optional[Union[torch.Tensor, np.ndarray]]  # (seq_len,)
    uuid: Optional[str] = None

    @classmethod
    def collate_fn(cls, sample_list: list["T2PSample"]):
        # TODO: rewrite this as a tree map
        batch = T2PSample(
            bboxes=torch.from_numpy(np.stack([s.bboxes for s in sample_list])),
            bodies=torch.from_numpy(np.stack([s.bodies for s in sample_list])),
            faces=torch.from_numpy(np.stack([s.faces for s in sample_list])),
            hands=torch.from_numpy(np.stack([s.hands for s in sample_list])),
            pose_mask=torch.from_numpy(np.stack([s.pose_mask for s in sample_list])),
            image_ratio=torch.from_numpy(np.stack([s.image_ratio for s in sample_list])),
            input_ids=torch.from_numpy(
                np.stack(
                    [s.input_ids for s in sample_list],
                    dtype=np.int32,
                )
            )
            if all(s.input_ids is not None for s in sample_list)
            else None,  # type: ignore
            uuid=[s.uuid for s in sample_list],
        ).to(dtype=torch.float32)
        return batch

    @classmethod
    def split_batch(cls, batch: "T2PSample", micro_batch_size: int) -> list["T2PSample"]:
        return [
            T2PSample(
                bboxes=batch.bboxes[i * micro_batch_size : (i + 1) * micro_batch_size],
                bodies=batch.bodies[i * micro_batch_size : (i + 1) * micro_batch_size],
                faces=batch.faces[i * micro_batch_size : (i + 1) * micro_batch_size],
                hands=batch.hands[i * micro_batch_size : (i + 1) * micro_batch_size],
                pose_mask=batch.pose_mask[i * micro_batch_size : (i + 1) * micro_batch_size],
                image_ratio=batch.image_ratio[i * micro_batch_size : (i + 1) * micro_batch_size],
                input_ids=batch.input_ids[i * micro_batch_size : (i + 1) * micro_batch_size]
                if batch.input_ids is not None
                else None,
                uuid=batch.uuid[i * micro_batch_size : (i + 1) * micro_batch_size]
                if batch.uuid is not None
                else None,
            )
            for i in range(cls.get_batch_size(batch) // micro_batch_size)
        ]

    @classmethod
    def get_batch_size(cls, batch: "T2PSample") -> int:
        return len(batch.image_ratio)

    def to(
        self,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        # TODO: rewrite this as a tree map
        return T2PSample(
            bboxes=_to_device(self.bboxes, device, dtype),
            bodies=_to_device(self.bodies, device, dtype),
            faces=_to_device(self.faces, device, dtype),
            hands=_to_device(self.hands, device, dtype),
            pose_mask=_to_device(self.pose_mask, device, dtype),
            image_ratio=_to_device(self.image_ratio, device, dtype),
            input_ids=_to_device(self.input_ids, device),
            uuid=self.uuid,
        )

    def numpy(self):
        return T2PSample(
            bboxes=_to_numpy(self.bboxes),
            bodies=_to_numpy(self.bodies),
            faces=_to_numpy(self.faces),
            hands=_to_numpy(self.hands),
            pose_mask=_to_numpy(self.pose_mask),
            image_ratio=_to_numpy(self.image_ratio),
            input_ids=_to_numpy(self.input_ids),
            uuid=self.uuid,
        )

    def squeeze(self, dim: int = 0):
        return T2PSample(
            bboxes=self.bboxes.squeeze(dim),
            bodies=self.bodies.squeeze(dim),
            faces=self.faces.squeeze(dim),
            hands=self.hands.squeeze(dim),
            pose_mask=self.pose_mask.squeeze(dim),
            image_ratio=self.image_ratio.squeeze(dim),
            input_ids=self.input_ids.squeeze(dim) if self.input_ids is not None else None,
            uuid=self.uuid,
        )

    def unsqueeze(self, dim: int = 0):
        return T2PSample(
            bboxes=_unsqueeze(self.bboxes, dim),
            bodies=_unsqueeze(self.bodies, dim),
            faces=_unsqueeze(self.faces, dim),
            hands=_unsqueeze(self.hands, dim),
            pose_mask=_unsqueeze(self.pose_mask, dim),
            image_ratio=_unsqueeze(self.image_ratio, dim),
            input_ids=_unsqueeze(self.input_ids, dim) if self.input_ids is not None else None,
            uuid=self.uuid,
        )
