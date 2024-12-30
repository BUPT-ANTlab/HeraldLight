from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from peft import PeftModel

mode_path = '/root/autodl-tmp/HuggingFace-Download-Accelerator/llama3.1/models--meta-llama--Meta-Llama-3.1-8B-Instruct'
lora_path = '/root/autodl-tmp/finetune_test/jinan3/checkpoint-708'
save_path = '/root/autodl-tmp/finetune_test/jinan3/merged'

tokenizer = AutoTokenizer.from_pretrained(mode_path, trust_remote_code=True)

base_model = AutoModelForCausalLM.from_pretrained(mode_path, device_map="auto", torch_dtype=torch.bfloat16, trust_remote_code=True).eval()

lora_model = PeftModel.from_pretrained(base_model, model_id=lora_path)

merged_model = lora_model.merge_and_unload()

merged_model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)
