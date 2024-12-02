import torch
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype=torch.float32, trust_remote_code=True
        ).to(args.device)
