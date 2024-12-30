import pandas as pd
import json
def transform_data(input_data):
    instruction = input_data["prompt"][0]["content"][0]["text"]
    output = input_data["answer"]
    transformed = {
        "instruction": instruction,
        "input": "",
        "output": output
    }
    return transformed


if  __name__ == "__main__":
    data_list = []
    with open('jinan2_Llama_log_220.json', 'r') as f:
        data = json.load(f)
    for i,j in enumerate(data):
        data_list.append(transform_data(j))

    with open('jinan2_Llama_log_220.json_formatted.json', 'w') as json_file:
        json.dump(data_list, json_file, indent=4)
