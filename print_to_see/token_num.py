from transformers import AutoTokenizer


tokenizer = AutoTokenizer.from_pretrained("/root/autodl-tmp/HuggingFace-Download-Accelerator/llama3.1/models--meta-llama--Meta-Llama-3.1-8B-Instruct")

with open("output.txt", "r", encoding="utf-8") as file:
    content = file.read()
print(content)
tokens = tokenizer.encode(content, add_special_tokens=True)

token_count = len(tokens)

print(f"输入文本的token数量: {token_count}")
