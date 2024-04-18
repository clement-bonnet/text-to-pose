import os
from typing import Literal, Optional

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions import (
    Bernoulli,
    Categorical,
    Distribution,
    MixtureSameFamily,
    MultivariateNormal,
)
from transformers import CLIPTextModel
from huggingface_hub import PyTorchModelHubMixin

from t2p.t2p_sample import T2PSample
from t2p.types import PoseBodiesDict, PoseDict
from t2p.sampling import tempered_sampling
from t2p.transformer import TransformerLayer


class T2PTransformer(nn.Module, PyTorchModelHubMixin):
    def __init__(
        self,
        num_layers: int = 4,
        num_heads: int = 6,
        embed_dim_per_head: int = 64,
        mlp_dim_factor: int = 4,
        dropout: float = 0.0,
        max_num_poses: int = 5,
        cross_embed_dim: int = 768,
        predict_bodies: bool = True,
        predict_faces: bool = True,
        predict_hands: bool = True,
        distribution: Literal["gmm", "gaussian"] = "gmm",
        gmm_num_components: int = 6,
        clip_text_model: Optional[CLIPTextModel] = None,
        clip_text_model_device: Optional[torch.device] = None,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.embed_dim_per_head = embed_dim_per_head
        self.embed_dim = num_heads * embed_dim_per_head
        self.num_heads = num_heads
        self.mlp_dim_factor = mlp_dim_factor
        self.dropout = dropout
        self.max_num_poses = max_num_poses
        self.predict_bodies = predict_bodies
        self.predict_faces = predict_faces
        self.predict_hands = predict_hands
        assert distribution in ["gaussian", "gmm"]
        self.distribution = distribution
        self.clip_text_model = clip_text_model or CLIPTextModel.from_pretrained(
            "openai/clip-vit-large-patch14",
        )
        for p in self.clip_text_model.parameters():
            p.requires_grad = False
        self.clip_text_model.eval().to(clip_text_model_device)

        self.one_pose_seq_length = 18 * predict_bodies + 68 * predict_faces + 42 * predict_hands
        # 1 for start token, 2 for each bounding box, `one_pose_seq_length` for each pose
        self.max_seq_len = 1 + 2 * max_num_poses + self.one_pose_seq_length * max_num_poses
        self.start_image_ratio_embedding = nn.Linear(1, self.embed_dim)
        self.bbox_linear = nn.Linear(2, self.embed_dim)
        self.bbox_xy_embedding = nn.Embedding(2, self.embed_dim)
        if predict_bodies:
            self.bodies_linear = nn.Linear(2, self.embed_dim)
            self.bodies_embedding = nn.Embedding(18, self.embed_dim)
        if predict_faces:
            self.faces_linear = nn.Linear(2, self.embed_dim)
            self.faces_embedding = nn.Embedding(68, self.embed_dim)
        if predict_hands:
            self.hands_linear = nn.Linear(2, self.embed_dim)
            self.hands_embedding = nn.Embedding(42, self.embed_dim)
        self.person_positional_embedding = nn.Embedding(max_num_poses, self.embed_dim)
        self.transformer_layers = nn.ModuleList(
            [
                TransformerLayer(
                    self.embed_dim,
                    self.num_heads,
                    self.dropout,
                    self.mlp_dim_factor,
                    cross_embed_dim,
                )
                for _ in range(self.num_layers)
            ]
        )
        self.last_layer_norm = nn.LayerNorm(self.embed_dim)
        if self.distribution == "gaussian":
            self.num_logits = 1 + 5
        elif self.distribution == "gmm":
            self.gmm_num_components = gmm_num_components
            self.num_logits = 1 + gmm_num_components + 5 * gmm_num_components
        self.linear_head_bboxes = nn.Linear(
            self.embed_dim,
            self.num_logits,
        )
        self.linear_head_bodies = nn.Linear(
            self.embed_dim,
            self.num_logits,
        )
        self.linear_head_faces = nn.Linear(
            self.embed_dim,
            self.num_logits,
        )
        self.linear_head_hands = nn.Linear(
            self.embed_dim,
            self.num_logits,
        )

    def forward(
        self, batch: T2PSample, text_embeddings: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        batch = batch.to(self.clip_text_model.device)
        if text_embeddings is None and batch.input_ids is not None:
            with torch.no_grad():
                text_embeddings = self.clip_text_model(
                    batch.input_ids, output_hidden_states=True
                ).hidden_states[
                    -2
                ]  # type: ignore

        embeddings, padding_mask = self._embed_pose(batch)  # (bs, t, h) with t = 1 + 128*n
        for transformer_layer in self.transformer_layers:
            embeddings = transformer_layer(
                embeddings,
                text_embeddings,
                padding_mask,
            )
        embeddings = self.last_layer_norm(embeddings)
        return self._project_embeddings(embeddings)

    def loss(
        self,
        logits: torch.Tensor,
        inputs: T2PSample,
        binary_labels_eps: float = 1e-2,
    ) -> dict[str, torch.Tensor]:
        inputs = inputs.to(logits.device)
        logits = logits[:, :-1]  # drop logit of last token
        bs = logits.shape[0]
        bbox_labels = inputs.bboxes.reshape(bs, -1, 2)
        pose_labels_list = []
        if self.predict_bodies:
            pose_labels_list.append(inputs.bodies)
        if self.predict_faces:
            pose_labels_list.append(inputs.faces)
        if self.predict_hands:
            pose_labels_list.append(inputs.hands)
        if pose_labels_list:
            pose_labels = torch.cat(pose_labels_list, dim=-2)  # (bs, n, 128, 2)
            pose_labels = pose_labels.reshape(bs, -1, 2)  # (bs, 128*n, 2)
            labels = torch.cat([bbox_labels, pose_labels], dim=-2)
        else:
            labels = bbox_labels
        binary_labels = (labels != -1).all(dim=-1).float()
        binary_distribution, distribution = self._logits_to_distribution(logits)
        # Binary mask is True for non-negative bboxes, the first negative bbox point, and all pose points
        # that correspond to a non-negative bbox point
        binary_mask = binary_labels.bool()
        binary_mask[:, 2 * self.max_num_poses :] = (
            inputs.pose_mask[:, :, None].repeat(1, 1, self.one_pose_seq_length).reshape(bs, -1)
        )
        next_bbox_indices = binary_mask.float().argmin(dim=-1).clamp(max=2 * self.max_num_poses - 1)
        binary_mask[torch.arange(bs), next_bbox_indices] = True
        # Remove odd indices from binary_mask to only keep the first bbox point
        binary_mask[:, 1 : 2 * self.max_num_poses : 2] = False

        binary_losses = F.binary_cross_entropy_with_logits(
            binary_distribution.logits, binary_labels, reduction="none"
        )
        binary_loss = torch.where(binary_mask, binary_losses, 0).sum() / (
            binary_mask.sum() + binary_labels_eps
        )

        log_likelihood = distribution.log_prob(labels)
        neg_log_likelihood = torch.where(binary_labels.bool(), -log_likelihood, 0).sum() / (
            binary_labels.sum() + binary_labels_eps
        )

        loss = binary_loss + neg_log_likelihood
        metrics = self._compute_loss_metrics(
            loss=loss,
            binary_loss=binary_loss,
            neg_log_likelihood=neg_log_likelihood,
            distribution=distribution,
            binary_mask=binary_mask,
            binary_labels=binary_labels,
            binary_losses=binary_losses,
            log_likelihood=log_likelihood,
            binary_labels_eps=binary_labels_eps,
        )
        return {"total": loss, **metrics}

    def eval_forward(self, batch: T2PSample, outputs: None = None) -> dict[str, torch.Tensor]:
        del outputs
        logits = self(batch)
        metrics = self.loss(logits, batch)
        return metrics

    @torch.no_grad()
    def _compute_loss_metrics(
        self,
        loss: torch.Tensor,
        binary_loss: torch.Tensor,
        neg_log_likelihood: torch.Tensor,
        distribution: Distribution,
        binary_mask: torch.Tensor,
        binary_labels: torch.Tensor,
        binary_losses: torch.Tensor,
        log_likelihood: torch.Tensor,
        binary_labels_eps: float,
    ) -> dict[str, torch.Tensor]:
        pose_length = self.one_pose_seq_length
        binary_loss_bboxes = torch.where(
            binary_mask[:, : 2 * self.max_num_poses],
            binary_losses[:, : 2 * self.max_num_poses],
            0,
        ).sum() / (binary_mask[:, : 2 * self.max_num_poses].sum() + binary_labels_eps)
        neg_log_likelihood_bboxes = torch.where(
            binary_labels[:, : 2 * self.max_num_poses].bool(),
            -log_likelihood[:, : 2 * self.max_num_poses],
            0,
        ).sum() / (binary_labels[:, : 2 * self.max_num_poses].sum() + binary_labels_eps)

        if self.predict_bodies:
            binary_loss_bodies, neg_log_likelihood_bodies = 0, 0
            for i in range(self.max_num_poses):
                start = pose_length * i + 2 * self.max_num_poses
                end = pose_length * i + 2 * self.max_num_poses + 18
                binary_loss_bodies += torch.where(
                    binary_mask[:, start:end],
                    binary_losses[:, start:end],
                    0,
                ).sum() / (binary_mask[:, start:end].sum() + binary_labels_eps)
                neg_log_likelihood_bodies += torch.where(
                    binary_labels[:, start:end].bool(),
                    -log_likelihood[:, start:end],
                    0,
                ).sum() / (binary_labels[:, start:end].sum() + binary_labels_eps)
        else:
            binary_loss_bodies, neg_log_likelihood_bodies = None, None

        if self.predict_faces:
            binary_loss_faces, neg_log_likelihood_faces = 0, 0
            for i in range(self.max_num_poses):
                start = pose_length * i + 2 * self.max_num_poses + 18
                end = pose_length * i + 2 * self.max_num_poses + 18 + 68
                binary_loss_faces += torch.where(
                    binary_mask[:, start:end],
                    binary_losses[:, start:end],
                    0,
                ).sum() / (binary_mask[:, start:end].sum() + binary_labels_eps)
                neg_log_likelihood_faces += torch.where(
                    binary_labels[:, start:end].bool(),
                    -log_likelihood[:, start:end],
                    0,
                ).sum() / (binary_labels[:, start:end].sum() + binary_labels_eps)
        else:
            binary_loss_faces, neg_log_likelihood_faces = None, None

        if self.predict_hands:
            binary_loss_hands, neg_log_likelihood_hands = 0, 0
            for i in range(self.max_num_poses):
                start = pose_length * i + 2 * self.max_num_poses + 18 + 68
                end = pose_length * i + 2 * self.max_num_poses + 18 + 68 + 42
                binary_loss_hands += torch.where(
                    binary_mask[:, start:end],
                    binary_losses[:, start:end],
                    0,
                ).sum() / (binary_mask[:, start:end].sum() + binary_labels_eps)
                neg_log_likelihood_hands += torch.where(
                    binary_labels[:, start:end].bool(),
                    -log_likelihood[:, start:end],
                    0,
                ).sum() / (binary_labels[:, start:end].sum() + binary_labels_eps)
        else:
            binary_loss_hands, neg_log_likelihood_hands = None, None

        metrics = {
            "binary_loss": binary_loss,
            "neg_log_likelihood": neg_log_likelihood,
            "loss": loss,
            "binary_loss_bboxes": binary_loss_bboxes,
            "neg_log_likelihood_bboxes": neg_log_likelihood_bboxes,
            "binary_loss_bodies": binary_loss_bodies,
            "neg_log_likelihood_bodies": neg_log_likelihood_bodies,
            "binary_loss_faces": binary_loss_faces,
            "neg_log_likelihood_faces": neg_log_likelihood_faces,
            "binary_loss_hands": binary_loss_hands,
            "neg_log_likelihood_hands": neg_log_likelihood_hands,
        }
        if self.distribution == "gmm":
            metrics.update(mixture_entropy=distribution.mixture_distribution.entropy().mean())
        return metrics

    def _project_embeddings(self, embeddings: torch.Tensor) -> torch.Tensor:
        logits_bboxes = self.linear_head_bboxes(embeddings)  # (bs, t, 6) if Gaussian
        logits_bodies = self.linear_head_bodies(embeddings)  # (bs, t, 6) if Gaussian
        logits_faces = self.linear_head_faces(embeddings)  # (bs, t, 6) if Gaussian
        logits_hands = self.linear_head_hands(embeddings)  # (bs, t, 6) if Gaussian

        concat_list = []
        pose_length = self.one_pose_seq_length
        bodies_offset = 18 * self.predict_bodies
        faces_offset = 68 * self.predict_faces
        for i in range(self.max_num_poses):
            if self.predict_bodies:
                start = pose_length * i + 2 * self.max_num_poses
                end = pose_length * i + 2 * self.max_num_poses + 18
                concat_list.append(logits_bodies[:, start:end])
            if self.predict_faces:
                start = pose_length * i + 2 * self.max_num_poses + bodies_offset
                end = pose_length * i + 2 * self.max_num_poses + bodies_offset + 68
                concat_list.append(logits_faces[:, start:end])
            if self.predict_hands:
                start = pose_length * i + 2 * self.max_num_poses + bodies_offset + faces_offset
                end = pose_length * i + 2 * self.max_num_poses + bodies_offset + faces_offset + 42
                concat_list.append(logits_hands[:, start:end])
        logits = torch.cat(
            [
                logits_bboxes[:, : 2 * self.max_num_poses],
                *concat_list,
                torch.zeros_like(logits_bboxes[:, 0:1]),  # last token
            ],
            dim=-2,
        )
        return logits

    def _embed_pose(self, inputs: T2PSample) -> tuple[torch.Tensor, torch.Tensor]:
        """Embeds pose inputs into a sequence of embeddings of shape (bs, 1 + 128*n, h)."""
        if inputs.bodies.ndim != 4:  # (bs, n, 18, 2)
            raise ValueError("Expected batched inputs.")

        # Image ratio
        image_ratios = inputs.image_ratio[:, None, None]  # (bs, 1, 1)
        start_emb = self.start_image_ratio_embedding(image_ratios)  # (bs, 1, h)

        # Bounding boxes
        bbox_emb = self.bbox_linear(inputs.bboxes)  # (bs, n, 2, h)
        # Bounding boxes positional embeddings (broadcasted to all bounding boxes)
        bbox_emb = bbox_emb + self.bbox_xy_embedding(torch.arange(0, 2, device=bbox_emb.device))
        # Add person positional embeddings
        person_embeddings = self.person_positional_embedding(
            torch.arange(0, self.max_num_poses, device=bbox_emb.device)
        )[
            None, :, None, :
        ]  # (1, n, 1, h)
        bbox_emb = bbox_emb + person_embeddings  # (bs, n, 2, h)
        bbox_emb = bbox_emb.reshape(-1, 2 * self.max_num_poses, self.embed_dim)  # (bs, 2*n, h)

        pose_emb_list = []
        # Bodies
        if self.predict_bodies:
            bodies_emb = self.bodies_linear(inputs.bodies)  # (bs, n, 18, h)
            # Bodies positional embeddings (broadcasted to all bodies)
            bodies_emb = bodies_emb + self.bodies_embedding(
                torch.arange(0, 18, device=bodies_emb.device)
            )
            pose_emb_list.append(bodies_emb)

        # Faces
        if self.predict_faces:
            faces_emb = self.faces_linear(inputs.faces)  # (bs, n, 68, h)
            # Faces positional embeddings (broadcasted to all faces)
            faces_emb = faces_emb + self.faces_embedding(
                torch.arange(0, 68, device=faces_emb.device)
            )
            pose_emb_list.append(faces_emb)

        # Hands
        if self.predict_hands:
            hands_emb = self.hands_linear(inputs.hands)  # (bs, n, 42, h)
            # Hands positional embeddings (broadcasted to all hands)
            hands_emb = hands_emb + self.hands_embedding(
                torch.arange(0, 42, device=hands_emb.device)
            )
            pose_emb_list.append(hands_emb)

        if pose_emb_list:
            # Concatenate all body parts
            pose_emb = torch.cat(pose_emb_list, dim=-2)  # (bs, n, 128, h)
            # Add person positional embeddings
            pose_emb = pose_emb + person_embeddings  # (bs, n, 128, h)

            pose_emb = pose_emb.reshape(
                -1, self.one_pose_seq_length * self.max_num_poses, self.embed_dim
            )  # (bs, 128*n, h)
            embeddings = torch.cat(
                [start_emb, bbox_emb, pose_emb], dim=-2
            )  # (bs, 1 + 2*n + 128*n, h)
        else:
            embeddings = torch.cat([start_emb, bbox_emb], dim=-2)  # (bs, 1 + 2*n, h)

        padding_mask = torch.zeros_like(embeddings[..., 0], dtype=torch.bool)  # (bs, t)
        # Start token
        padding_mask[:, 0] = True
        # Bounding boxes
        pose_mask = torch.all(inputs.bboxes != -1.0, dim=-1).any(-1)[:, :, None]
        padding_mask[:, 1 : 1 + 2 * self.max_num_poses] = pose_mask.repeat(1, 1, 2).reshape(
            -1, 2 * self.max_num_poses
        )
        # Poses
        if pose_emb_list:
            padding_mask[:, 1 + 2 * self.max_num_poses :] = pose_mask.repeat(
                1, 1, self.one_pose_seq_length
            ).reshape(-1, self.one_pose_seq_length * self.max_num_poses)

        return embeddings, padding_mask

    @torch.no_grad()
    def generate(
        self,
        text_embeddings: Optional[torch.Tensor] = None,
        num_poses: int = 1,
        generator: Optional[torch.Generator] = None,
        bbox_dist_temperature: float = 1.0,
        pose_dist_temperature: float = 1.0,
        bbox_cls_threshold: float = 0.5,
        pose_cls_threshold: float = 0.5,
        tempered_softmax_size: int = 1000,
        device: Optional[torch.device | str] = None,
        image_ratio: float = 1.0,
    ):
        """Generate poses from given text embeddings."""
        device = device or next(self.parameters()).device
        dtype = next(self.parameters()).dtype
        if text_embeddings is not None:
            assert (
                text_embeddings.ndim == 2
            ), "Expected text_embeddings to be of shape (time, hidden)."
            text_embeddings = text_embeddings[None].tile(num_poses, 1, 1).to(device)
        poses = T2PSample(
            bboxes=torch.full(
                (num_poses, self.max_num_poses, 2, 2),
                -1.0,
                device=device,
                dtype=dtype,
            ),
            bodies=torch.full(
                (num_poses, self.max_num_poses, 18, 2),
                -1.0,
                device=device,
                dtype=dtype,
            ),
            faces=torch.full(
                (num_poses, self.max_num_poses, 68, 2),
                -1.0,
                device=device,
                dtype=dtype,
            ),
            hands=torch.full(
                (num_poses, self.max_num_poses, 42, 2),
                -1.0,
                device=device,
                dtype=dtype,
            ),
            pose_mask=torch.zeros(num_poses, self.max_num_poses, device=device, dtype=torch.bool),
            image_ratio=torch.full(
                (num_poses,),
                image_ratio,
                device=device,
                dtype=dtype,
            ),
            input_ids=None,
            uuid=None,
        )
        bbox_before = torch.ones((num_poses,), device=device, dtype=torch.bool)
        for bbox_index in range(self.max_num_poses):
            if not bbox_before.any():
                break
            previous_poses = poses
            poses = self._add_new_bbox(
                poses,
                text_embeddings,
                bbox_index,
                generator,
                dist_temperature=bbox_dist_temperature,
                cls_threshold=bbox_cls_threshold,
                tempered_softmax_size=tempered_softmax_size,
            )
            poses = T2PSample(
                bboxes=torch.where(
                    bbox_before[:, None, None, None],
                    poses.bboxes,
                    previous_poses.bboxes,
                ),
                bodies=poses.bodies,
                faces=poses.faces,
                hands=poses.hands,
                pose_mask=torch.where(
                    bbox_before[:, None],
                    poses.pose_mask,
                    previous_poses.pose_mask,
                ),
                image_ratio=poses.image_ratio,
                input_ids=poses.input_ids,
                uuid=poses.uuid,
            )
            bbox_before = poses.pose_mask[:, bbox_index]
        # TODO: reorder bounding boxes by size in case they are not ordered.
        for pose_index in range(self.max_num_poses):
            if not poses.pose_mask[:, pose_index].any():
                break

            for pose_part_index in range(self.one_pose_seq_length):
                poses = self._add_new_point(
                    poses,
                    text_embeddings,
                    pose_index,
                    pose_part_index,
                    generator,
                    dist_temperature=pose_dist_temperature,
                    cls_threshold=pose_cls_threshold,
                    tempered_softmax_size=tempered_softmax_size,
                )
            poses = T2PSample(
                bboxes=poses.bboxes,
                bodies=torch.where(
                    poses.pose_mask[:, pose_index][:, None, None, None],
                    poses.bodies,
                    -1,
                ),
                faces=torch.where(
                    poses.pose_mask[:, pose_index][:, None, None, None],
                    poses.faces,
                    -1,
                ),
                hands=torch.where(
                    poses.pose_mask[:, pose_index][:, None, None, None],
                    poses.hands,
                    -1,
                ),
                pose_mask=poses.pose_mask,
                image_ratio=poses.image_ratio,
                input_ids=poses.input_ids,
                uuid=poses.uuid,
            )

        if num_poses == 1:
            poses = T2PSample(
                bboxes=poses.bboxes[0],
                bodies=poses.bodies[0],
                faces=poses.faces[0],
                hands=poses.hands[0],
                pose_mask=poses.pose_mask[0],
                image_ratio=poses.image_ratio[0],
                input_ids=poses.input_ids,
                uuid=poses.uuid,
            )
        return poses

    def _logits_to_distribution(
        self,
        logits: torch.Tensor,
        sigma_min: float = 1e-3,
    ) -> tuple[Bernoulli, Distribution]:
        batch_dims = logits.shape[:-1]
        cls_distribution = Bernoulli(logits=logits[..., 0])
        if self.distribution == "gaussian":
            assert logits.shape[-1] == 1 + 5
            scale_tril = torch.stack(
                [
                    sigma_min + F.softplus(logits[..., 3]),
                    torch.zeros_like(logits[..., 0]),
                    logits[..., 4],
                    sigma_min + F.softplus(logits[..., 5]),
                ],
                dim=-1,
            ).view(*batch_dims, 2, 2)
            distribution = MultivariateNormal(loc=logits[..., 1:3], scale_tril=scale_tril)
        elif self.distribution == "gmm":
            assert logits.shape[-1] == 1 + self.gmm_num_components + 5 * self.gmm_num_components
            categorical_logits = logits[..., 1 : self.gmm_num_components + 1]
            component_logits = logits[..., self.gmm_num_components + 1 :]
            loc = component_logits[..., : self.gmm_num_components * 2].view(
                *batch_dims, self.gmm_num_components, 2
            )
            coeff_a = F.softplus(
                component_logits[..., 2 * self.gmm_num_components : 3 * self.gmm_num_components]
            )
            coeff_b = torch.zeros_like(component_logits[..., : self.gmm_num_components])
            coeff_c = component_logits[
                ..., 3 * self.gmm_num_components : 4 * self.gmm_num_components
            ]
            coeff_d = F.softplus(
                component_logits[..., 4 * self.gmm_num_components : 5 * self.gmm_num_components]
            )
            scale_tril = torch.stack(
                [
                    sigma_min + coeff_a,
                    coeff_b,
                    coeff_c,
                    sigma_min + coeff_d,
                ],
                dim=-1,
            ).view(*batch_dims, self.gmm_num_components, 2, 2)
            distribution = MixtureSameFamily(
                mixture_distribution=Categorical(logits=categorical_logits),
                component_distribution=MultivariateNormal(loc=loc, scale_tril=scale_tril),
            )
        else:
            raise ValueError(f"Unknown distribution {self.distribution}.")
        return cls_distribution, distribution

    @torch.no_grad()
    def _add_new_bbox(
        self,
        poses: T2PSample,
        text_embeddings: Optional[torch.Tensor],
        bbox_index: int,
        generator: Optional[torch.Generator] = None,
        dist_temperature: float = 1.0,
        cls_threshold: float = 0.5,
        tempered_softmax_size: int = 1000,
    ) -> T2PSample:
        # First point of the bbox
        logits = self(poses, text_embeddings)[:, 2 * bbox_index]  # there are 2 points for each bbox
        cls_distribution, distribution = self._logits_to_distribution(logits)

        bbox_exists = cls_distribution.probs > cls_threshold
        if isinstance(distribution, MultivariateNormal):
            distribution._unbroadcasted_scale_tril *= dist_temperature
            # FIX: use generator to sample points
            bbox = distribution.sample()
        else:
            # GMM does not have mode or tempered sampling implemented, so we use approximate tempered sampling.
            bbox = tempered_sampling(
                distribution,
                temperature=dist_temperature,
                softmax_size=tempered_softmax_size,
            )
        bbox = bbox.clamp(0, 1)

        bboxes = poses.bboxes
        pose_mask = poses.pose_mask
        # TODO: make it work for a batch of points_exist instead of a for loop
        for b_i, bbox_exist in enumerate(bbox_exists):
            if bbox_exist:
                pose_mask[b_i, bbox_index] = True
                bboxes[b_i, bbox_index, 0] = bbox[b_i]
            else:
                pose_mask[b_i, bbox_index] = False

        poses = T2PSample(
            bboxes=bboxes,
            bodies=poses.bodies,
            faces=poses.faces,
            hands=poses.hands,
            pose_mask=pose_mask,
            image_ratio=poses.image_ratio,
            input_ids=poses.input_ids,
            uuid=poses.uuid,
        )

        # Second point of the bbox
        # TODO: maybe sample between (x1, y1) and (1, 1) instead of (0, 0) and (1, 1)
        logits = self(poses, text_embeddings)[:, 2 * bbox_index + 1]
        _, distribution = self._logits_to_distribution(logits)
        if isinstance(distribution, MultivariateNormal):
            distribution._unbroadcasted_scale_tril *= dist_temperature
            # FIX: use generator to sample points
            bbox = distribution.sample()
        else:
            # GMM does not have mode or tempered sampling implemented, so we use approximate tempered sampling.
            bbox = tempered_sampling(
                distribution, temperature=dist_temperature, softmax_size=tempered_softmax_size
            )
        bbox = bbox.clamp(0, 1)

        # TODO: make it work for a batch of points_exist instead of a for loop
        for b_i, bbox_exist in enumerate(pose_mask[:, bbox_index]):
            if bbox_exist:
                bboxes[b_i, bbox_index, 1] = bbox[b_i]

        return T2PSample(
            bboxes=bboxes,
            bodies=poses.bodies,
            faces=poses.faces,
            hands=poses.hands,
            pose_mask=pose_mask,
            image_ratio=poses.image_ratio,
            input_ids=poses.input_ids,
            uuid=poses.uuid,
        )

    @torch.no_grad()
    def _add_new_point(
        self,
        poses: T2PSample,
        text_embeddings: Optional[torch.Tensor],
        pose_index: int,
        pose_part_index: int,
        generator: Optional[torch.Generator] = None,
        dist_temperature: float = 1.0,
        cls_threshold: float = 0.5,
        tempered_softmax_size: int = 1000,
    ) -> T2PSample:
        bbox_offset = 2 * self.max_num_poses
        index = pose_index * self.one_pose_seq_length + pose_part_index
        logits = self(poses, text_embeddings)[:, bbox_offset + index]  # (bs, 6) if Gaussian
        cls_distribution, distribution = self._logits_to_distribution(logits)

        points_exist = cls_distribution.probs > cls_threshold
        if isinstance(distribution, MultivariateNormal):
            if dist_temperature == 0.0:
                points = distribution.mean
            else:
                distribution._unbroadcasted_scale_tril *= dist_temperature
                # FIX: use generator to sample points
                points = distribution.sample()
        else:
            # GMM does not have mode or tempered sampling implemented, so we use approximate tempered sampling.
            points = tempered_sampling(
                distribution, temperature=dist_temperature, softmax_size=tempered_softmax_size
            )
        points = points.clamp(0, 1)

        bodies = poses.bodies
        faces = poses.faces
        hands = poses.hands
        # TODO: make it work for a batch of points_exist instead of a for loop
        one_pose_length = self.one_pose_seq_length
        bodies_offset = 18 * self.predict_bodies
        faces_offset = 68 * self.predict_faces
        for b_i, point_exists in enumerate(points_exist):
            if point_exists:
                if self.predict_bodies:
                    if index % one_pose_length < 18:
                        bodies[b_i, index // one_pose_length, index % one_pose_length] = points[b_i]
                if self.predict_faces:
                    if bodies_offset <= index % one_pose_length < bodies_offset + 68:
                        faces[
                            b_i, index // one_pose_length, index % one_pose_length - bodies_offset
                        ] = points[b_i]
                if self.predict_hands:
                    if (
                        bodies_offset + faces_offset
                        <= index % one_pose_length
                        < bodies_offset + faces_offset + 42
                    ):
                        hands[
                            b_i,
                            index // one_pose_length,
                            index % one_pose_length - bodies_offset - faces_offset,
                        ] = points[b_i]

        return T2PSample(
            bboxes=poses.bboxes,
            bodies=bodies,
            faces=faces,
            hands=hands,
            pose_mask=poses.pose_mask,
            image_ratio=poses.image_ratio,
            input_ids=poses.input_ids,
            uuid=poses.uuid,
        )

    def convert_to_dwpose(
        self,
        pose: T2PSample,
    ) -> PoseDict | list[PoseDict]:
        """Converts a (batch of) T2PSample to a (list of) DWpose."""
        pose = pose.numpy()
        if pose.bodies.ndim == 4:
            dw_pose = []
            for i in range(pose.bodies.shape[0]):
                pose_i = T2PSample(
                    bboxes=pose.bboxes[i],
                    bodies=pose.bodies[i],
                    faces=pose.faces[i],
                    hands=pose.hands[i],
                    pose_mask=pose.pose_mask[i],
                    image_ratio=pose.image_ratio,
                    input_ids=pose.input_ids[i] if pose.input_ids is not None else None,
                    uuid=pose.uuid[i] if pose.uuid is not None else None,
                )
                dw_pose.append(self.convert_to_dwpose(pose_i))
        elif pose.bodies.ndim == 3:
            indices = np.arange(0, 18 * self.max_num_poses).reshape(
                self.max_num_poses,
                18,
            )
            bodies = pose.bodies
            subset = np.where((pose.bodies != -1.0).all(-1), indices, -1.0)
            faces = pose.faces
            hands = pose.hands
            # Rescale using bounding boxes
            bboxes = pose.bboxes
            bodies = np.where(
                bodies != -1,
                (bodies * (bboxes[:, 1:2] - bboxes[:, 0:1]) + bboxes[:, 0:1]),
                bodies,
            ).reshape(18 * self.max_num_poses, 2)
            faces = np.where(
                faces != -1,
                (faces * (bboxes[:, 1:2] - bboxes[:, 0:1]) + bboxes[:, 0:1]),
                faces,
            )
            hands = np.where(
                hands != -1,
                (hands * (bboxes[:, 1:2] - bboxes[:, 0:1]) + bboxes[:, 0:1]),
                hands,
            ).reshape(self.max_num_poses * 2, 21, 2)

            dw_pose = PoseDict(
                bodies=PoseBodiesDict(candidate=bodies, subset=subset),
                faces=faces,
                hands=hands,
            )
        else:
            raise ValueError(f"Expected single pose or batch of poses, got {pose}.")
        return dw_pose


if __name__ == "__main__":
    from PIL import Image
    from transformers import AutoProcessor

    from utils import draw_pose

    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    device = torch.device("cpu")
    t2p_transformer = T2PTransformer.from_pretrained("clement-bonnet/t2p-transformer-v0")
    t2p_transformer.to(device).eval()
    print(
        "Number of parameters: {:,}".format(
            sum(p.numel() for p in t2p_transformer.parameters() if p.requires_grad)
        )
    )

    prompt = "a lady dancing on a stage"
    num_poses = 3
    temperature = 0.1
    image_ratio = 1.0

    clip_processor = AutoProcessor.from_pretrained("openai/clip-vit-large-patch14")
    with torch.no_grad():
        input_ids = clip_processor(
            text=prompt, return_tensors="pt", padding="max_length", truncation=True
        )["input_ids"]
        text_embeddings = (
            t2p_transformer.clip_text_model(input_ids.to(device), output_hidden_states=True)
            .hidden_states[-2]
            .squeeze()
        )

    poses = t2p_transformer.generate(
        text_embeddings=text_embeddings,
        num_poses=num_poses,
        bbox_dist_temperature=temperature,
        pose_dist_temperature=temperature,
        image_ratio=image_ratio,
    )
    dw_poses = t2p_transformer.convert_to_dwpose(poses)
    for dw_pose in dw_poses:
        Image.fromarray(draw_pose(dw_pose, 1024, int(1024 * image_ratio))).show()
