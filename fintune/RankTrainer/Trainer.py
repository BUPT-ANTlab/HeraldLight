from transformers import AutoTokenizer, AutoModelForCausalLM
from RankTrainer import *
from datasets import load_dataset, Dataset
from peft import LoraConfig, get_peft_model
from collections import defaultdict

import random
from transformers import Seq2SeqTrainingArguments

model_name = '/root/autodl-tmp/finetune_test/hangzhou2/merged'

tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    padding_side="left",
    padding=True
)
tokenizer.pad_token_id = 0




model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype = torch.bfloat16,
    device_map = "auto",
)

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)

data = load_dataset("json", data_files=f"hangzhou2_error_corrected.json")

train_val = data["train"].train_test_split(test_size=0.1, shuffle=True, seed=2024)
train_data = train_val["train"].shuffle(seed=2024)
val_data = train_val["test"].shuffle(seed=2024)
model.train()

data_module = make_supervised_data_module(tokenizer, train_data, val_data, mix=False)

training_args = TrainingArguments(
        output_dir="./results",
        evaluation_strategy="steps",
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=3e-4,
        save_strategy="steps",
        num_train_epochs=3,
        logging_dir="./logs",
        logging_steps=1,
        model_max_length=2048
)

rank_trainer = RankTrainer(
    model=model,
    args=training_args,
    tokenizer=tokenizer,
    **data_module
)


rank_trainer.train()

output_model_dir = "./trained_model"
model.save_pretrained(output_model_dir)
tokenizer.save_pretrained(output_model_dir)

print(f"模型和tokenizer已保存到 {output_model_dir}")



from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from peft import PeftModel

mode_path = '/root/autodl-tmp/finetune_test/hangzhou2/merged'
lora_path = '/root/autodl-tmp/sychronize/fintune/RankTrainer/trained_model'
save_path = '/root/autodl-tmp/sychronize/fintune/RankTrainer/merged_model'


tokenizer = AutoTokenizer.from_pretrained(mode_path, trust_remote_code=True)


base_model = AutoModelForCausalLM.from_pretrained(mode_path, device_map="auto", torch_dtype=torch.bfloat16, trust_remote_code=True).eval()


lora_model = PeftModel.from_pretrained(base_model, model_id=lora_path)

merged_model = lora_model.merge_and_unload()

merged_model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)




