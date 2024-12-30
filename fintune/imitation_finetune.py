import os
import sys
from typing import List
import fire
import torch
from datasets import load_dataset
import transformers
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model



from transformers import LlamaForCausalLM, LlamaTokenizer, BitsAndBytesConfig, PreTrainedTokenizerFast

def train(
        # model/data params
        base_model: str = "/root/autodl-tmp/HuggingFace-Download-Accelerator/llama3.1/models--meta-llama--Meta-Llama-3.1-8B-Instruct",
        data_path: str = "/root/autodl-tmp/finetune_test/output.json",
        output_dir: str = "/root/autodl-tmp/finetune_test/adaptor",
        # training hyperparams
        batch_size: int = 32,
        micro_batch_size: int = 4,
        num_epochs: int = 3,
        learning_rate: float = 2e-5,
        cutoff_len: int = 2048,
        val_set_size: int = 0.05,
        # lora hyperparams
        lora_r: int = 8,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        lora_target_modules: List[str] = [
            "q_proj",
            "v_proj",
        ],
        # llm hyperparams
        train_on_inputs: bool = True,
        group_by_length: bool = True,
        # other
        mask: bool = False,
):
    print(f"Training Llama 3.1 model with params:\n"
          f"base_model: {base_model}\n"
          f"data_path: {data_path}\n"
          f"output_dir: {output_dir}\n"
          f"batch_size: {batch_size}\n"
          f"micro_batch_size: {micro_batch_size}\n"
          f"num_epochs: {num_epochs}\n"
          f"learning_rate: {learning_rate}\n"
          f"cutoff_len: {cutoff_len}\n"
          f"val_set_size: {val_set_size}\n"
          f"lora_r: {lora_r}\n"
          f"lora_alpha: {lora_alpha}\n"
          f"lora_dropout: {lora_dropout}\n"
          f"lora_target_modules: {lora_target_modules}\n"
          f"train_on_inputs: {train_on_inputs}\n"
          f"group_by_length: {group_by_length}\n")

    gradient_accumulation_steps = batch_size // micro_batch_size
    device_map = "auto"
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if ddp:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
        gradient_accumulation_steps = gradient_accumulation_steps // world_size

    quantization_config = BitsAndBytesConfig(load_in_8bit=True)

    model = LlamaForCausalLM.from_pretrained(
        base_model,
        quantization_config=quantization_config,
        device_map=device_map,
        rope_scaling={"type": "dynamic", "factor": 8.0}
    )

    tokenizer = PreTrainedTokenizerFast.from_pretrained(base_model)

    tokenizer.pad_token_id = 0
    tokenizer.padding_side = "left"

    def tokenize(prompt, add_eos_token=True):
        result = tokenizer(
            prompt,
            truncation=True,
            max_length=cutoff_len,
            padding=False,
            return_tensors=None
        )

        if (
                result["input_ids"][-1] != tokenizer.eos_token_id
                and len(result["input_ids"]) < cutoff_len
                and add_eos_token
        ):
            result["input_ids"].append(tokenizer.eos_token_id)
            result["attention_mask"].append(1)
        result["labels"] = result["input_ids"].copy()
        return result

    def generate_and_tokenize_prompt(data_point):
        full_prompt = generate_prompt(data_point)
        tokenized_full_prompt = tokenize(full_prompt)
        if not train_on_inputs:
            user_prompt = generate_prompt({**data_point, "output": ""})
            tokenized_user_prompt = tokenize(user_prompt, add_eos_token=False)
            user_prompt_len = len(tokenized_user_prompt["input_ids"])

            tokenized_full_prompt["labels"] = [
                                                  -100
                                              ] * user_prompt_len + tokenized_full_prompt["labels"][
                                                                    user_prompt_len:
                                                                    ]
        return tokenized_full_prompt

    model = prepare_model_for_kbit_training(model)

    config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=lora_target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, config)

    data = load_dataset("json", data_files=data_path)

    if val_set_size > 0:
        train_val = data["train"].train_test_split(
            test_size=val_set_size, shuffle=True, seed=42
        )
        train_data = train_val["train"].shuffle().map(generate_and_tokenize_prompt)
        val_data = train_val["test"].shuffle().map(generate_and_tokenize_prompt)
    else:
        train_data = data["train"].shuffle().map(generate_and_tokenize_prompt)
        val_data = None

    trainer = transformers.Trainer(
    model=model,
    train_dataset=train_data,
    eval_dataset=val_data,
    args=transformers.TrainingArguments(
        per_device_train_batch_size=micro_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        warmup_steps=100,
        num_train_epochs=num_epochs,
        learning_rate=learning_rate,
        fp16=True,
        logging_steps=10,
        evaluation_strategy="steps" if val_set_size > 0 else "no",
        save_strategy="steps",
        eval_steps=200 if val_set_size > 0 else None,
        save_steps=400,
        output_dir=output_dir,
        save_total_limit=3,
        load_best_model_at_end=True if val_set_size > 0 else False,
        ddp_find_unused_parameters=False if ddp else None,
        group_by_length=group_by_length,
        gradient_checkpointing=True,  # 启用梯度检查点
    ),
    data_collator=transformers.DataCollatorForSeq2Seq(
        tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
    ),
)


    model.config.use_cache = False

    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    trainer.train()
    model.save_pretrained(output_dir)

    print("\nTraining completed.")

def generate_prompt(data_point):
    return f"""### Instruction:
{data_point["instruction"]}

### Response:
{data_point["output"]}"""

if __name__ == "__main__":
    fire.Fire(train)
