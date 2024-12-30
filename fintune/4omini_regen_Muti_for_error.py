import json
from tqdm import tqdm
import wandb
import os
import sys
import time
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
import re

def llama3170B(prompt):
    OPENROUTER_API_KEY = ""
    YOUR_SITE_URL = ""
    YOUR_APP_NAME = "TSCD"
    max_retries = 5
    retry_delay = 60

    for attempt in range(max_retries):
        try:
            response = requests.post(
                url="https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                    "HTTP-Referer": f"{YOUR_SITE_URL}",
                    "X-Title": f"{YOUR_APP_NAME}",
                },
                data=json.dumps({
                    "model": "openai/gpt-4o-mini",
                    "messages": [prompt[0][1]],
                    "top_p": 1,
                    "temperature": 0.7,
                    "frequency_penalty": 0,
                    "presence_penalty": 0,
                    "repetition_penalty": 1,
                    "top_k": 0,
                    "provider": {
                        "order": [
                            "OpenAI",
                        ]
                    },
                })
            )
            response.raise_for_status()
            response_data = response.json()
            text_content = response_data['choices'][0]['message']['content']
            return text_content

        except requests.exceptions.Timeout:
            print(f"Timeout error occurred on attempt {attempt + 1}. Retrying in {retry_delay} seconds...")

        except requests.exceptions.ConnectionError:
            print(f"Connection error occurred on attempt {attempt + 1}. Retrying in {retry_delay} seconds...")

        except requests.exceptions.HTTPError as http_err:
            print(f"HTTP error occurred on attempt {attempt + 1}: {http_err}. Retrying in {retry_delay} seconds...")

        except requests.exceptions.RequestException as err:
            print(f"An error occurred on attempt {attempt + 1}: {err}. Retrying in {retry_delay} seconds...")

        except ValueError as parse_err:
            print(f"Failed to parse JSON on attempt {attempt + 1}: {parse_err}. Retrying in {retry_delay} seconds...")

        if attempt < max_retries - 1:
            time.sleep(retry_delay)
        else:
            print(f"Failed after {max_retries} attempts. Please check your connection or API status.")
            return None


data = []
try:
    with open('hangzhou2_error.json', 'r', encoding='utf-8') as file:
        data = json.load(file)
except json.JSONDecodeError as e:
    print(f"JSON Decode Error: {e}")
except FileNotFoundError:
    print("文件 'synthetic_prompts.json' 未找到。")
except Exception as e:
    print(f"加载数据时发生未预期的错误: {e}")


def prepare_messages(prompt):

    messages = [
        {
            "role": "system",
            "content": "you are a master in traffic signal control",
        },
        {
            "role": "user",
            "content":prompt['error_prompt'][0][0]['content'][0]['text']
                }
    ]
    return messages


def process_prompt(index, prompt, lock, f):

    messages = prepare_messages(prompt)



    output = llama3170B([messages])

    piece = {'query': prompt['error_prompt'][0][0]['content'][0]['text'], 'responses':[output, prompt['error_answer']], "scores": [0.8, 0.2]}

    with lock:
        if index != 0:
            f.write(",\n")
        json.dump(piece, f, ensure_ascii=False, indent=4)
    return index


def custom_Llama_inference(model, tokenizer, sampling_params, syn_prompts):
    messages_list = []
    print("len(prompts)", len(syn_prompts))
    for prompt in syn_prompts:
        messages = prepare_messages(prompt)
        messages_list.append(messages)
    print('len(messages_list)', len(messages_list))

    with open('hangzhou2_error_corrected.json', 'w', encoding='utf-8') as f:
        f.write("[\n")

        wandb.log({'total_step': len(messages_list)})

        lock = Lock()
        results = []
        with ThreadPoolExecutor(max_workers=10) as executor:
            future_to_index = {executor.submit(process_prompt, i, syn_prompts[i], lock, f): i for i in
                               range(len(messages_list))}

            for future in tqdm(as_completed(future_to_index), total=len(messages_list), desc="Generating responses"):
                index = future_to_index[future]
                try:
                    result_index = future.result()
                    wandb.log({'now_step': result_index})
                except Exception as exc:
                    print(f"Prompt {index} generated an exception: {exc}")

        f.write("\n]")

    print(f"JSON file saved as 'output.json'")


wandb.init(project="synthetic", name="dynamiclight")

custom_Llama_inference(None, None, None, data)

wandb.finish()
