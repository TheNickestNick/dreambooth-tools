import argparse
import torch
from torch import autocast
from diffusers import StableDiffusionPipeline

def parse_args():
    parser = argparse.ArgumentParser(description='Diffusion Inference')
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--prompt', type=str, required=True)
    parser.add_argument('--num_inference_steps', type=int, default=1)
    parser.add_argument('--guidance_scale', type=float, default=1)
    parser.add_argument("--mixed_precision", type=str, default="no",
        choices=["no", "fp16", "bf16"],)
    parser.add_argument("--output", type=str, default="output.png")
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

    with autocast("cuda"):
        image = pipe(args.prompt,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale).images[0]

    image.save(args.output)

if __name__ == '__main__':
    main()