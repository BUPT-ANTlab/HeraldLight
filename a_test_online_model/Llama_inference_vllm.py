from vllm import LLM, SamplingParams
import torch
from transformers import AutoTokenizer


def load_Llama(model_path, diverse_generation):
    a =  LLM(model_path, max_model_len=(5120), trust_remote_code=True, max_num_batched_tokens=(5120), #1289
               tensor_parallel_size=1, enforce_eager=False, dtype=torch.float16, tokenizer_mode = "auto")
    b = a.llm_engine.tokenizer.tokenizer

    if diverse_generation:
        sampling_params = SamplingParams(temperature=1.0,
                                         top_p=0.9,
                                         max_tokens=1024,
                                         stop_token_ids=[b.eos_token_id,
                                                         b.convert_tokens_to_ids("<|eot_id|>")])
    else:
        sampling_params = SamplingParams(temperature=0.1,
                                         top_p=0.9,
                                         max_tokens=1024,
                                         stop_token_ids=[b.eos_token_id,
                                                         b.convert_tokens_to_ids("<|eot_id|>")])
    return a, b, sampling_params


def Llama_inference(model, tokenizer, sampling_params, prompts):
    #tokenizer = model.llm_engine.tokenizer.tokenizer
    messages_list = []
    print("len(prompts)", len(prompts))
    for prompt in prompts:
        messages = [
            {
                "role": "system",
                "content": "you are a master in traffic signal control",
            },
            {
                "role": "user",
                "content": prompt[0]['content'][0]['text'],
            }
        ]
        messages_list.append(messages)
    print('len(messages_list)',len(messages_list))
    formatted_prompts = [
        tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        for messages in messages_list
    ]
    # template = """
	# <|begin_of_text|><|start_header_id|>system<|end_header_id|>
	# {system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>
	# {user_msg_1}<|eot_id|>
	# """.format(system_prompt="you are a traffic control master", user_msg_1=messages[1]['content'])

    output = model.generate(
        formatted_prompts,
        sampling_params,

    )
    print("len formatted_prompts", len(formatted_prompts))
    #print(output)
    a = []
    for i in range(len(formatted_prompts)):
        generated_text = output[i].outputs[0].text
        a.append({'content': generated_text})
    return a


if __name__ == "__main__":
    model_path = '/root/autodl-tmp/finetune_test/4omini236/merged'

    # Load the model with vllm
    a, b, c = load_Llama(model_path, False)

    prompts = [[{'content' : [{'text' : "Intersection Knowledge:\nThis intersection operates with a four-signal-phase system. The signal phases are defined as follows:\nETWT (East and West Through): Permits vehicles to proceed straight in both the East and West directions.\nNTST (North and South Through): Permits vehicles to proceed straight in both the North and South directions.\nELWL (East and West Left-Turn): Permits vehicles to make left turns in both the East and West directions.\nNLSL (North and South Left-Turn): Permits vehicles to make left turns in both the North and South directions.\nTask Description:\nTask 1: Signal Phase Selection\nYou will receive the queueing vehicle data for each of the four signal phases. Your task is to select the most urgent phase based on the following criteria:\n1. Total Queue Calculation:\nEmpty: Indicates no vehicles are queued for that phase.\n[num1, num2]: Represents the number of queued vehicles in two lanes controlled by the phase. For example, ETWT controls one lane in the East and one in the West. Sum `num1` and `num2` to obtain the total queue for the phase.\n[num1]: Indicates that only one lane has queued vehicles, and the other lane is empty.\n2. Phase Comparison:\nCompare the total queue numbers across all four phases.\nIf queue totals are similar between phases, assess the balance of each phase:\nA large difference between `num1` and `num2` signifies an imbalance. An imbalanced phase leads to inefficient use of traffic duration, as the signal allows both lanes to proceed simultaneously, potentially wasting time when one lane has significantly fewer queued vehicles.\n3. Two Versions of Queueing Set:\nYou will see two sets of queueing vehicles for each phase: the Herald version and the Original version.\nThe Herald version calculates future vehicle movements, effectively representing the queueing situation. It is a highly efficient prediction method, and in most cases, it is recommended to use this version for decision-making.\nThe Original version represents the vehicles queuing at the current time step and does not account for vehicles that will be running in the future. Only when the Herald version shows severe imbalance(for example the final duration calculated based on Herald Version is too long (The duration > 30), you should analyze if this long duration is reasonable) is it recommended to use the Original version to achieve short-term gains.\nTask 2: Duration Selection\nAfter selecting the optimal signal phase in Task 1, determine the appropriate traffic duration using the following steps:\n1. Identify the larger number between `num1` and `num2` in the selected phase and denote it as A.\n2. Calculate the initial duration:\nDuration = (A * 3) - 1\n3. Adjust the duration based on the following rules (If you picked Original version, this step should be skipped):\nIf `Duration \u2265 20`, then `Duration = Duration - 3`.\nIf `Duration = 14`, then `Duration = Duration - 2`.\nTask Details:\nThe queueing numbers for each phase are provided as follows:\nHerald version: \nETWT: Empty\nNTST: Empty\nELWL: [3]\nNLSL: Empty\nOriginal version: \nETWT: Empty\nNTST: Empty\nELWL: [3]\nNLSL: Empty\nRequirements:\nStep-by-Step Reasoning: Provide a detailed analysis following these steps:\n1. Identify the Optimal Traffic Signal: Analyze the queueing data to select the most urgent phase.\n2. Calculate the Duration: Determine the appropriate duration based on the selected phase.\n3. Provide the Final Decision: Present the chosen signal phase and duration.\nSelection Constraints:\nOnly one signal phase can be selected.\nThe final answer must be formatted precisely as:\n<signal>YOUR_CHOICE</signal>, <duration>YOUR_CHOICE</duration>\n*Example*: <signal>ETWT</signal>, <duration>5</duration>\nAdditional Rules:\nIf all signal phases are empty, select any signal phase with a default duration of 5 to keep the intersection operational.\nEnsure the duration is within the range of 0 to 40. Durations outside this range are considered invalid.\nEach tag (`<signal>` and `<duration>`) should appear only once in the final answer.\nImportant:\nYou MUST provide the answer in the specified format: `<signal>YOUR_CHOICE</signal>, <duration>YOUR_CHOICE</duration>`. Any other format will not be accepted.\n",


                              }]}]]
    d  = Llama_inference(a, b, c, prompts)
    print(d)
