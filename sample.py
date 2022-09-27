import argparse
from pathlib import Path
import torch
from torch import autocast
from diffusers import StableDiffusionPipeline
from datetime import datetime

def timestamp():
    return datetime.now().strftime("%Y%m%d%H%M%S")

def parse_args():
    parser = argparse.ArgumentParser(description='Diffusion Inference')
    parser.add_argument('--base_output_dir', type=str, default='output')
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--prompt', type=str, required=True)
    parser.add_argument('--num_inference_steps', type=int, default=1)
    parser.add_argument('--guidance_scale', type=float, default=1)
    parser.add_argument("--mixed_precision", type=str, default="no",
        choices=["no", "fp16", "bf16"],)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_batches", type=int, default=1)
    args = parser.parse_args()
    return args

def main():
    args = parse_args()

    dtype = torch.float32
    if args.mixed_precision == "fp16":
        dtype = torch.float16
    elif args.mixed_precision == "bf16":
        dtype = torch.bfloat16

    pipe = StableDiffusionPipeline.from_pretrained(
        args.model_path, torch_dtype=dtype).to("cuda")

    output_dir = Path(args.base_output_dir) / timestamp()
    output_dir.mkdir(parents=True, exist_ok=True)

    with autocast("cuda"):
        image_counter = 0
        for _ in range(args.num_batches):
            images = pipe([args.prompt] * args.batch_size,
                          num_inference_steps=args.num_inference_steps,
                          guidance_scale=args.guidance_scale).images
            for img in images:
                img.save(output_dir / f"{image_counter:04d}.png")

if __name__ == '__main__':
    main()