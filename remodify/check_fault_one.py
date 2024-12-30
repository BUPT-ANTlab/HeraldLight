import json
import re
from a_test_online_model.mini_request import llama3170B
from utils.my_utils import append_to_json_file

with open('error.json', 'r', encoding='utf-8') as file:
    data = json.load(file)


phases = ['ETWT', 'NTST', 'ELWL', 'NLSL']




import re

def extract_all_task_details(text, phases):

    task_pattern = re.compile(
        r"Task Details:\s*The queueing numbers for each phase are provided as follows:\s*Herald version:\s*(.*?)(?=\nOriginal version:|$)",
        re.DOTALL
    )

    tasks = task_pattern.findall(text)

    all_queue_numbers = []

    for task_content in tasks:
        queue_numbers = {}
        for phase in phases:
            # 构造每个相位的正则表达式模式
            # 匹配形式：[num] 或 [num1, num2] 或 Empty
            pattern = rf"{phase}:\s*(Empty|\[([^\]]+)\])"
            match = re.search(pattern, task_content)
            if match:
                if match.group(1) == "Empty":
                    queue_numbers[phase] = 0
                elif match.group(2):
                    numbers = [int(num.strip()) for num in match.group(2).split(',')]
                    queue_numbers[phase] = numbers
            else:
                queue_numbers[phase] = 0
        all_queue_numbers.append(queue_numbers)

    return all_queue_numbers


def regenerate(prompt):
    answer = llama3170B(prompt)
    return answer
wrong1 = 0
wrong2 = 0

collection = []
for index, piece in enumerate(data):
    #print(index)
    if 'prompt' in piece:
        queue_data = extract_all_task_details(piece['prompt'][0]['content'][0]['text'], phases)
        #print(piece['prompt'][0]['content'][0]['text'])
        queue_data = queue_data[0]

        origin_queue = queue_data[phases[piece['phase_action']]]

        large_number = max([origin_queue] if origin_queue == 0 else origin_queue)

        duration_need = large_number * 3 - 1

        if duration_need >= 20:
            duration_need -= 3
        elif duration_need == 14:
            duration_need -= 2

        if duration_need == -1:
            duration_need = 5




        if not (duration_need == piece['duration_action']):
            temp_dict = {"query": "", "responses": [], "scores": [0.2, 0.8]}
            print(f"1. {queue_data}, 2. {origin_queue}, 3. {large_number}, 4. {piece['duration_action']}, 5. {duration_need}")
            print("\n", piece)
            wrong2 += 1
            # temp_dict["query"] = piece['prompt'][0]['content'][0]['text']
            # temp_dict["responses"].append(piece['answer']['content'])
            # temp_dict["responses"].append(regenerate(piece['prompt']))
            #
            # append_to_json_file("fixed.json",temp_dict)

        elif large_number == 0 :
            if piece['duration_action'] != 5:
                print(f"1. {queue_data}, 2. {origin_queue}, 3. {large_number}, 4. {piece['duration_action']}")

                wrong1 += 1

print('data length:', len(data), 'wrong_times', wrong1, wrong2)




