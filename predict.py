from cog import BasePredictor, Input, Path
from typing import List
import torch
from PIL import Image
from insightface.app import FaceAnalysis
from ip_adapter.ip_adapter_faceid import IPAdapterFaceID
from diffusers import StableDiffusionPipeline, DDIMScheduler, AutoencoderKL
import os

class Predictor(BasePredictor):
    def setup(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(self.device)
        self.pipe = StableDiffusionPipeline.from_pretrained(
            "SG161222/Realistic_Vision_V5.1_noVAE",
            torch_dtype=torch.float16,
            scheduler=DDIMScheduler.from_pretrained("SG161222/Realistic_Vision_V5.1_noVAE", subfolder="scheduler"),
            safety_checker=None,
            feature_extractor=None,
            requires_safety_checker=False,
            vae=self.vae
        ).to(self.device)

        self.face_encoder = FaceAnalysis(name="buffalo_l", providers=["CUDAExecutionProvider" if self.device == "cuda" else "CPUExecutionProvider"])
        self.face_encoder.prepare(ctx_id=0 if self.device == "cuda" else -1)

        self.ip_model = IPAdapterFaceID(pipe=self.pipe, model_ckpt="h94/IP-Adapter-FaceID", device=self.device)
        self.pipe.set_progress_bar_config(disable=True)

    def predict(self, face_image: Path = Input(description="Image with one face"), prompt: str = Input(default="a cinematic photo of {}"), num_samples: int = Input(default=1)) -> List[Path]:
        image = Image.open(face_image).convert("RGB")
        faceid_embeds = self.ip_model.get_face_embed(image)
        prompt = prompt.format("a person")

        images = self.ip_model.generate(
            prompt=prompt,
            negative_prompt="",
            num_images=num_samples,
            faceid_embeds=faceid_embeds,
            scale=0.8,
            num_inference_steps=30,
            seed=42,
        )

        output_paths = []
        for i, img in enumerate(images):
            out_path = f"/tmp/output_{i}.png"
            img.save(out_path)
            output_paths.append(Path(out_path))

        return output_paths
