from transformers import PreTrainedTokenizerFast

def count_tokens(text: str, model_name: str = "path_or_model_name"):

    tokenizer = PreTrainedTokenizerFast.from_pretrained(model_name)


    tokens = tokenizer(text, return_tensors="pt")

    num_tokens = len(tokens['input_ids'][0])

    print(f"文本的 token 数量为: {num_tokens}")
    return num_tokens

text = ""
count_tokens(text)
