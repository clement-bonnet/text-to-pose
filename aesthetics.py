import torch
from torch import nn
from PIL import Image
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection


class MLP(nn.Module):
    def __init__(self, input_size: int):
        super().__init__()
        self.input_size = input_size
        layers = [
            nn.Linear(self.input_size, 1024),
            nn.Dropout(0.2),
            nn.Linear(1024, 128),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.Dropout(0.1),
            nn.Linear(64, 16),
            nn.Linear(16, 1),
        ]
        self.layers = nn.Sequential(*layers)
        self.clip_image_processor = CLIPImageProcessor()
        self.clip_encoder = CLIPVisionModelWithProjection.from_pretrained(
            "openai/clip-vit-large-patch14"
        ).eval()

    def forward(self, x):
        return self.layers(x)

    @torch.no_grad()
    def score(self, image: Image) -> float:
        transformed_images = self.clip_image_processor(
            images=image, return_tensors="pt"
        ).pixel_values
        clip_features = self.clip_encoder(transformed_images).image_embeds
        clip_features = clip_features / clip_features.norm(dim=-1, keepdim=True)
        return self(clip_features).item()


def get_aesthetic_classifier(
    weights_path: str = "sac+logos+ava1-l14-linearMSE.pth", device: str = "cpu"
) -> MLP:
    aesthetic_classifier = MLP(input_size=768)
    aesthetic_classifier.load_state_dict(torch.load(weights_path, map_location="cpu"), strict=False)
    aesthetic_classifier.eval().to(device)
    return aesthetic_classifier


if __name__ == "__main__":
    aesthetic_classifier = get_aesthetic_classifier()
    print(aesthetic_classifier.score(Image.open("knn_poses/pose_0.png")))
