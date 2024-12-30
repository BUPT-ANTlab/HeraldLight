import transformers
import torch


def load_Llama(model_path):
    model_id = model_path
    return transformers.pipeline(
        "text-generation",
        model=model_id,
        model_kwargs={"torch_dtype": torch.bfloat16},
        device_map="auto",
    )

def Llama_inference(pipeline, prompt):
    messages = [
        {"role": "system", "content": "You are a master in traffic signal control"},
        {"role": "user", "content": prompt[0]['content'][0]['text']},
    ]

    outputs = pipeline(
        messages,
        max_new_tokens=2048,
    )

    return outputs[0]["generated_text"][-1]

if __name__ == "__main__":
    model_path = '/root/autodl-tmp/HuggingFace-Download-Accelerator/modelDownload/models--meta-llama--Llama-2-7b-chat-hf'

    pipeline = load_Llama(model_path)

    c = {'content': {'text': "tell me what is green light"}}
    a = Llama_inference(pipeline, c)
