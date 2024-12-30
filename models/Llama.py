"""
DynamicLight under feature fusion method 1
Input shape: [batch, max_lane*4]
Created by Liang Zhang
"""

import random
import time
import re
import os
from utils.my_utils import *
from a_test_online_model.Llama_inference_vllm import *
from collections import deque
import math

four_phase_list = {'ETWT': 0, 'NTST': 1, 'ELWL': 2, 'NLSL': 3}
four_phase_list_reverse = {0:'ETWT', 1:'NTST', 2:'ELWL', 3:'NLSL'}
eight_phase_list = {'ETWT': 0, 'NTST': 1, 'ELWL': 2, 'NLSL': 3, 'WTWL': 4, 'ETEL': 5, 'STSL': 6, 'NTNL': 7}
location_dict = {"N": "North", "S": "South", "E": "East", "W": "West"}
location_dict_detail = {"N": "Northern", "S": "Southern", "E": "Eastern", "W": "Western"}
direction_dict = {"T": "through", "L": "left-turn", "R": "turn-right"}
direction_dict_ori = {"T": "through", "L": "turn-left", "R": "turn-right"}

phase_explanation_dict_detail = {"NTST": "- NTST: Northern and southern through lanes.",
                                 "NLSL": "- NLSL: Northern and southern left-turn lanes.",
                                 "NTNL": "- NTNL: Northern through and left-turn lanes.",
                                 "STSL": "- STSL: Southern through and left-turn lanes.",
                                 "ETWT": "- ETWT: Eastern and western through lanes.",
                                 "ELWL": "- ELWL: Eastern and western left-turn lanes.",
                                 "ETEL": "- ETEL: Eastern through and left-turn lanes.",
                                 "WTWL": "- WTWL: Western through and left-turn lanes."
                                }

incoming_lane_2_outgoing_road = {
    "NT": "South",
    "NL": "East",
    "ST": "North",
    "SL": "West",
    "ET": "West",
    "EL": "South",
    "WT": "East",
    "WL": "North"
}


class Llama(object):
    def __init__(self, dic_agent_conf, dic_traffic_env_conf, dic_path, cnt_round, intersection_id, log_dir,dataset ):
        self.dic_agent_conf = dic_agent_conf
        self.dic_traffic_env_conf = dic_traffic_env_conf
        self.dic_path = dic_path
        self.cnt_round = cnt_round
        self.intersection_id = intersection_id
        self.phases = four_phase_list
        current_time = time.strftime("%Y%m%d_%H%M%S", time.localtime())
        self.state_action_prompt_file = f"{log_dir}/{dataset}_state_action_prompt_domain_knowledge_{current_time}.json"
        self.error_file = f"{log_dir}/{dataset}_error_prompts_domain_knowledge{current_time}.json"
        self.state_action_prompt = []
        self.first_time_to_select = 1
        self.errors =[]
        self.diverse_generation = False
        self.Llama_model, self.tokenizer , self.sampling_params = load_Llama(self.dic_agent_conf['Llama_path'], self.diverse_generation)

        self.relieve_history_of_all = deque(maxlen=30)
        self.relieve_history_of_all_num = deque(maxlen=30)
        self.now_choice_prompt = []
        self.four_states_prompt = []

    def choose_action(self, states, list_need, list_intersection, list_gpt_history, phase_action_plug_in, step_time):

        pactions = []
        dactions = []
        prompts = []
        self.four_states_prompt = []

        for i,intersection_index in enumerate(list_need):
            four_states_prompt =[]
            state = states[i]
            state_text, state_text_dict = self.state_to_text(state)
            four_states_prompt.append(state_text_dict)
            relieve_history, one_piece_history_for_expert, one_piece_history_for_expert_num ="", "", ""

            if relieve_history == None:
                relieve_history = ""

            self.relieve_history_of_all.append(one_piece_history_for_expert)
            self.relieve_history_of_all_num.append(one_piece_history_for_expert_num)

            phase_sums = {}
            phase_early_queued = {}
            queued_time_dict = list_intersection[intersection_index].dic_vehicle_queued_time_in

            #herald计算
            current_vehicles_per_lane = list_intersection[intersection_index].dic_lane_vehicle_current_step
            current_waiting_vechicles_num_per_lane_in = list_intersection[intersection_index].dic_lane_waiting_vehicle_count_current_step
            current_vehicles_speed_per_vehicle = list_intersection[intersection_index].dic_vehicle_speed_current_step
            current_vehicles_distance_per_vehicle = list_intersection[intersection_index].dic_vehicle_distance_current_step
            lane_length = list_intersection[intersection_index].lane_length

            queue_snap_no_herald, queue_snap_origin_no_herald = self.calculate_phase_queue_sums_v2(
                current_waiting_vechicles_num_per_lane_in)
            updated_waiting = self.heraldmaker_v2(current_vehicles_per_lane, current_waiting_vechicles_num_per_lane_in, current_vehicles_speed_per_vehicle, current_vehicles_distance_per_vehicle, lane_length)

            queue_snap, queue_snap_origin = self.calculate_phase_queue_sums_v2(updated_waiting)
            print('queue_snap, queue_snap_origin', queue_snap, queue_snap_origin)

            if queue_snap_origin != []:
                max_a, max_b = self.choose_max_sublist(queue_snap_origin[0])
                print("max_a, max_b", max_a, max_b)

            prompts.append(self.get_prompt_v2(queue_snap_origin[0] if queue_snap_origin != [] else [], queue_snap_origin_no_herald[0] if queue_snap_origin_no_herald != [] else []))


            if self.first_time_to_select :
                if queue_snap_origin != [] and max_b != []:
                    paction = max_a

                    daction = max(max_b) * 3 - 1
                    if daction == 32:
                        daction -= 1
                    elif daction >= 20:
                        daction -= 3
                    elif daction == 14:
                        daction -= 2

                    if daction > 40:
                        daction = 40
                    print(f"{paction, daction}")
                else:
                    paction = random.randint(0, 3)
                    daction = 5

                pactions.append(paction)
                dactions.append(daction)

            self.four_states_prompt.append(four_states_prompt)
        print('len(self.four_states_prompt)', len(self.four_states_prompt))

        if not self.first_time_to_select :
            answers = Llama_inference(self.Llama_model,self.tokenizer, self.sampling_params, prompts)
            for i, answer in enumerate(answers):
                answer = answer['content']
                try_times = 0
                while True:
                    signal_answer_pattern_phase = r'<signal>(.*?)</signal>'
                    matches_phase = re.findall(signal_answer_pattern_phase, answer)

                    signal_answer_pattern_duration = r'<duration>(.*?)</duration>'
                    matches_duration = re.findall(signal_answer_pattern_duration, answer)


                    if matches_phase and matches_duration :
                        signal_text_phase = matches_phase[-1]
                        signal_text_duration = matches_duration[-1]


                        if (signal_text_phase in four_phase_list) and (signal_text_duration in self.dic_traffic_env_conf['ACTION_DURATION_gpt']):
                            print("accepted_debug:",signal_text_phase, signal_text_duration)
                            break
                        else:
                            print("error_debug:", signal_text_phase, signal_text_duration)
                            self.append_to_json_file(self.error_file,
                                                     {"error_prompt": prompts, "error_answer": answer})

                            reform_prompt = prompts[i]
                            reform_prompt[0]['content'][0]['text'] = reform_prompt[0]['content'][0]['text']

                            answer = \
                            Llama_inference(self.Llama_model, self.tokenizer, self.sampling_params, [reform_prompt])[0][
                                'content']

                    else:
                        print("No <signal> or <duration> tag found in the generated answer, regenerating...")

                        self.append_to_json_file(self.error_file,
                                                 {"error_prompt": prompts, "error_answer": answer})

                        reform_prompt = prompts[i]
                        reform_prompt[0]['content'][0]['text'] = reform_prompt[0]['content'][0]['text']

                        answer = Llama_inference(self.Llama_model,self.tokenizer, self.sampling_params, [reform_prompt])[0]['content']



                signal_index_phase = four_phase_list[signal_text_phase]
                signal_text_duration = self.dic_traffic_env_conf['ACTION_DURATION_gpt'][signal_text_duration]
                pactions.append(signal_index_phase)
                dactions.append(signal_text_duration)

            print('len(prompts)',len(prompts))
            for i in range(len(prompts)):
                json_state_i = {}
                json_state_i = numpy_serialization(states[i])

                ons_slice = {"state": json_state_i, "prompt": prompts[i], "answer": answers[i], "phase_action": pactions[i], "duration_action" :dactions[i], "Maxa" : max_a, "Maxb": max_b}
                # ons_slice = {"state": states[i], "prompt": prompts[i], "answer": answers[i], "phase_action": pactions[i], "duration_action" :dactions[i]}

                # self.state_action_prompt.append(
                #     {"state": states[i], "prompt": prompts[i], "answer": answers[i], "phase_action": pactions[i], "duration_action" :dactions[i]})
                #print('self.state_action_prompt', self.state_action_prompt)
                self.append_to_json_file(self.state_action_prompt_file, ons_slice)
                #dump_json(self.state_action_prompt, self.state_action_prompt_file)

        self.first_time_to_select = 0
        self.now_choice_prompt = list(zip(pactions, dactions))
        return pactions, dactions, {"history": self.relieve_history_of_all, "states_list": self.four_states_prompt, "now_choice_list": self.now_choice_prompt,'history_num':self.relieve_history_of_all_num}

    def get_prompt_v2(self, arrays, arrays_no_herald):
        if arrays == []:
            arrays = [[], [], [], []]

        if arrays_no_herald == []:
            arrays_no_herald = [[], [], [], []]

        prompt = [{"role": "user",
                   "content": [{"type": "text",
                                "text":
                                    "Intersection Knowledge:\n"
                                    "This intersection operates with a four-signal-phase system. The signal phases are defined as follows:\n"
                                    "ETWT (East and West Through): Permits vehicles to proceed straight in both the East and West directions.\n"
                                    "NTST (North and South Through): Permits vehicles to proceed straight in both the North and South directions.\n"
                                    "ELWL (East and West Left-Turn): Permits vehicles to make left turns in both the East and West directions.\n"
                                    "NLSL (North and South Left-Turn): Permits vehicles to make left turns in both the North and South directions.\n"

                                    "Task Description:\n"
                                    "Task 1: Signal Phase Selection\n"
                                    "You will receive the queueing vehicle data for each of the four signal phases. Your task is to select the most urgent phase based on the following criteria:\n"

                                    "1. Total Queue Calculation:\n"
                                    "Empty: Indicates no vehicles are queued for that phase.\n"
                                    "[num1, num2]: Represents the number of queued vehicles in two lanes controlled by the phase. For example, ETWT controls one lane in the East and one in the West. Sum `num1` and `num2` to obtain the total queue for the phase.\n"
                                    "[num1]: Indicates that only one lane has queued vehicles, and the other lane is empty.\n"

                                    "2. Phase Comparison:\n"
                                    "Compare the total queue numbers across all four phases.\n"
                                    "If queue totals are similar between phases, assess the balance of each phase:\n"
                                    "A large difference between `num1` and `num2` signifies an imbalance. An imbalanced phase leads to inefficient use of traffic duration, as the signal allows both lanes to proceed simultaneously, potentially wasting time when one lane has significantly fewer queued vehicles.\n"

                                    "3. Two Versions of Queueing Set:\n"
                                    "You will see two sets of queueing vehicles for each phase: the Herald version and the Original version.\n"
                                    "The Herald version calculates future vehicle movements, effectively representing the queueing situation. It is a highly efficient prediction method, and in most cases, it is recommended to use this version for decision-making.\n"
                                    "The Original version represents the vehicles queuing at the current time step and does not account for vehicles that will be running in the future. Only when the Herald version shows severe imbalance(for example the final duration calculated based on Herald Version is too long (The duration > 30), you should analyze if this long duration is reasonable) is it recommended to use the Original version to achieve short-term gains.\n"


                                    "Task 2: Duration Selection\n"

                                    "After selecting the optimal signal phase in Task 1, determine the appropriate traffic duration using the following steps:\n"

                                    "1. Identify the larger number between `num1` and `num2` in the selected phase and denote it as A.\n"
                                    "2. Calculate the initial duration:\n"
                                    "Duration = (A * 3) - 1\n"

                                    "3. Adjust the duration based on the following rules (If you picked Original version, this step should be skipped):\n"
                                    "If `Duration ≥ 20`, then `Duration = Duration - 3`.\n"
                                    "If `Duration = 14`, then `Duration = Duration - 2`.\n"

                                    "Task Details:\n"

                                    "The queueing numbers for each phase are provided as follows:\n"

                                    "Herald version: \n"

                                    f"ETWT: {arrays[0] if arrays[0] != [] else 'Empty'}\n"
                                    f"NTST: {arrays[1] if arrays[1] != [] else 'Empty'}\n"
                                    f"ELWL: {arrays[2] if arrays[2] != [] else 'Empty'}\n"
                                    f"NLSL: {arrays[3] if arrays[3] != [] else 'Empty'}\n"

                                    "Original version: \n"

                                    f"ETWT: {arrays_no_herald[0] if arrays_no_herald[0] != [] else 'Empty'}\n"
                                    f"NTST: {arrays_no_herald[1] if arrays_no_herald[1] != [] else 'Empty'}\n"
                                    f"ELWL: {arrays_no_herald[2] if arrays_no_herald[2] != [] else 'Empty'}\n"
                                    f"NLSL: {arrays_no_herald[3] if arrays_no_herald[3] != [] else 'Empty'}\n"

                                    "Requirements:\n"

                                    "Step-by-Step Reasoning: Provide a detailed analysis following these steps:\n"
                                    "1. Identify the Optimal Traffic Signal: Analyze the queueing data to select the most urgent phase.\n"
                                    "2. Calculate the Duration: Determine the appropriate duration based on the selected phase.\n"
                                    "3. Provide the Final Decision: Present the chosen signal phase and duration.\n"

                                    "Selection Constraints:\n"
                                    "Only one signal phase can be selected.\n"
                                    "The final answer must be formatted precisely as:\n"
                                    "<signal>YOUR_CHOICE</signal>, <duration>YOUR_CHOICE</duration>\n"
                                    "*Example*: <signal>ETWT</signal>, <duration>5</duration>\n"

                                    "Additional Rules:\n"
                                    "If all signal phases are empty, select any signal phase with a default duration of 5 to keep the intersection operational.\n"
                                    "Ensure the duration is within the range of ** 0 to 40 **. Durations outside this range are considered invalid.\n"
                                    "Each tag (`<signal>` and `<duration>`) should appear only once in the final answer.\n"

                                    "Important:\n"
                                    "**You MUST provide the answer in the specified format: `<signal>YOUR_CHOICE</signal>, <duration>YOUR_CHOICE</duration>`. Any other format will not be accepted.**\n"

                                }]

                   }]
        return prompt

    def get_prompt(self,state_text,relieve_history, dynamic_plug_in_choice, step_time):

        prompt = [{"role": "user",
                   "content": [{"type": "text",
                                "text": "step_time: " + str(step_time) +"\n"
                                        + relieve_history +
                                        "A crossroad connects two roads: the north-south and east-west. The traffic light is located at "
                                        "the intersection of the two roads. The north-south road is divided into two sections by the intersection: "
                                        "the north and south. Similarly, the east-west road is divided into the east and west. Each section "
                                        "has two lanes: a through and a left-turn lane. Each lane is further divided into three segments. "
                                        "Segment 1 is the closest to the intersection. Segment 2 is in the middle. Segment 3 is the farthest. "
                                        "In a lane, there may be early queued vehicles and approaching vehicles traveling in different segments. "
                                        "Early queued vehicles have arrived at the intersection and await passage permission. Approaching "
                                        "vehicles will arrive at the intersection in the future.\n\n"
                                        "The traffic light has 4 signal phases. Each signal relieves vehicles' flow in the group of two "
                                        "specific lanes. The state of the intersection is listed below. It describes:\n"
                                        "- The group of lanes relieving vehicles' flow under each traffic light phase.\n"
                                        "- The number of early queued vehicles of the allowed lanes of each signal.\n"
                                        "- The number of approaching vehicles in different segments of the allowed lanes of each signal.\n\n"
                                        + state_text +
                                        "Please answer:\n"
                                        "Which is the most effective traffic signal that will most significantly improve the traffic "
                                        "condition during the next phase, which relieves vehicles' flow of the allowed lanes of the signal?"
                                        "How long the duration will be the most optimal choice? (please choose the duration between 10 and 40)\n\n"
                                        "Note:\n"
                                        "The traffic congestion is primarily dictated by the early queued vehicles, with the MOST significant "
                                        "impact. You MUST pay the MOST attention to lanes with long queue lengths. It is NOT URGENT to "
                                        "consider vehicles in distant segments since they are unlikely to reach the intersection soon.\n\n"
                                        "Requirements:\n"
                                        "- Let's think step by step.\n"
                                        "- You can only choose one of the signals listed above.\n"
                                        "- You must follow the following steps to provide your analysis: Step 1: Provide your analysis "
                                        "for identifying the optimal traffic signal. Step 2: Answer your chosen signal.\n"
                                        "- Your choice can only be given after finishing the analysis.\n"
                                        "- Your choice must be identified by the tag: <signal>YOUR_CHOICE</signal>,<duration>YOUR_CHOICE</duration>."}]
                   }]


        return prompt
    def state_to_text(self,state):
        state_text = ""
        state_text_dict = {}
        phase_total = state["phase_total"][:12]
        lane_queue_vehicle_in = state["lane_queue_vehicle_in"][:12]
        lane_run_in_part = state["lane_run_in_part"][:12]
        num_in_deg = state["num_in_deg"][:12*4]
        num_in_deg_sliced = [num_in_deg[i:i + 4] for i in range(0, len(num_in_deg), 4)]

        phase_map = self.dic_traffic_env_conf['PHASE_MAP']

        for p, index in self.phases.items():
            lane_1 = p[:2]
            lane_2 = p[2:]

            queue_len_1 = int(lane_queue_vehicle_in[phase_map[index][0]])
            queue_len_2 = int(lane_queue_vehicle_in[phase_map[index][1]])

            seg_1_lane_1 = num_in_deg_sliced[phase_map[index][0]][0]
            seg_2_lane_1 = num_in_deg_sliced[phase_map[index][0]][1]
            seg_3_lane_1 = num_in_deg_sliced[phase_map[index][0]][2] + num_in_deg_sliced[phase_map[index][0]][3]

            seg_1_lane_2 = num_in_deg_sliced[phase_map[index][1]][0]
            seg_2_lane_2 = num_in_deg_sliced[phase_map[index][1]][1]
            seg_3_lane_2 = num_in_deg_sliced[phase_map[index][1]][2] + num_in_deg_sliced[phase_map[index][1]][3]

            state_text += (f"Signal: {p}\n"
                          f"Allowed lanes: {phase_explanation_dict_detail[p][8:-1]}\n"
                          f"- Early queued: {queue_len_1} ({location_dict[lane_1[0]]}), {queue_len_2} ({location_dict[lane_2[0]]}), {queue_len_1 + queue_len_2} (Total)\n"
                          f"- Segment 1: {seg_1_lane_1} ({location_dict[lane_1[0]]}), {seg_1_lane_2} ({location_dict[lane_2[0]]}), {seg_1_lane_1 + seg_1_lane_2} (Total)\n"
                          f"- Segment 2: {seg_2_lane_1} ({location_dict[lane_1[0]]}), {seg_2_lane_2} ({location_dict[lane_2[0]]}), {seg_2_lane_1 + seg_2_lane_2} (Total)\n"
                          f"- Segment 3: {seg_3_lane_1} ({location_dict[lane_1[0]]}), {seg_3_lane_2} ({location_dict[lane_2[0]]}), {seg_3_lane_1 + seg_3_lane_2} (Total)\n\n")

            state_text_dict[p] = (f"Signal: {p}\n"
                          f"Allowed lanes: {phase_explanation_dict_detail[p][8:-1]}\n"
                          f"- Early queued: {queue_len_1} ({location_dict[lane_1[0]]}), {queue_len_2} ({location_dict[lane_2[0]]}), {queue_len_1 + queue_len_2} (Total)\n"
                          f"- Segment 1: {seg_1_lane_1} ({location_dict[lane_1[0]]}), {seg_1_lane_2} ({location_dict[lane_2[0]]}), {seg_1_lane_1 + seg_1_lane_2} (Total)\n"
                          f"- Segment 2: {seg_2_lane_1} ({location_dict[lane_1[0]]}), {seg_2_lane_2} ({location_dict[lane_2[0]]}), {seg_2_lane_1 + seg_2_lane_2} (Total)\n"
                          f"- Segment 3: {seg_3_lane_1} ({location_dict[lane_1[0]]}), {seg_3_lane_2} ({location_dict[lane_2[0]]}), {seg_3_lane_1 + seg_3_lane_2} (Total)\n\n")

        return state_text, state_text_dict

    def relieve_waiting_vehicles_history(self, intersection_index, list_gpt_history):
        return self.state_to_text_for_history(list_gpt_history[intersection_index])


    def state_to_text_for_history(self, single_inter_history):
        total_text = ""
        one_piece_history_for_expert = ""
        one_piece_history_for_expert_num = []
        history_explain = (
            "At this intersection, there are three historical decision points that recorded the selected phase information and the resulting number of vehicles passing through. \n"
            "These three past states and decisions can serve as a reference for choosing the phase action and its duration for current decisions. \n"
            "Please study and understand these historical actions. \n"
            "Note: You are not required to mimic the historical actions. They only represent the objective outcome of vehicle flow caused by those decisions. \n"
            "Instead, focus on analyzing the number of vehicles released by these actions, and evaluate whether the historically chosen phase actions and durations."
            "There may be:"
            "1. newly unrecorded vehicles run and passed the intersection during the duration, "
            "2. The vehicles in Segment 3 didn't have enough time to pass during the duration."
            "So the The passed vehicle number may be not always equal to the total of all segments. \n"
            "effectively relieved the pressure on the segments.\n"
            "Here are three historical decision points :\n"
        )
        total_text += history_explain
        for i, history in enumerate(single_inter_history):

            if history == [0]:
                continue
            only_phase_state_text = ""
            phase_total = history[2]["phase_total"][:12]
            lane_queue_vehicle_in = history[2]["lane_queue_vehicle_in"][:12]
            lane_run_in_part = history[2]["lane_run_in_part"][:12]
            num_in_deg = history[2]["num_in_deg"][:12*4]
            num_in_deg_sliced = [num_in_deg[i:i + 4] for i in range(0, len(num_in_deg), 4)]

            phase_map = self.dic_traffic_env_conf['PHASE_MAP']


            total_text = self.three_text(i, history, lane_queue_vehicle_in,num_in_deg_sliced, phase_map, total_text)

            one_piece_history_for_expert, one_piece_history_for_expert_num = self.one_text(i, history, lane_queue_vehicle_in,num_in_deg_sliced, phase_map, one_piece_history_for_expert, one_piece_history_for_expert_num)
            #print("one_piece_history_for_expert",one_piece_history_for_expert)
        return total_text, one_piece_history_for_expert, one_piece_history_for_expert_num

    def three_text(self, i, history, lane_queue_vehicle_in,num_in_deg_sliced, phase_map, total_text):

        for p, index in self.phases.items():
            if index == history[0] and i == 0:
                lane_1 = p[:2]
                lane_2 = p[2:]

                queue_len_1 = int(lane_queue_vehicle_in[phase_map[index][0]])
                queue_len_2 = int(lane_queue_vehicle_in[phase_map[index][1]])

                seg_1_lane_1 = num_in_deg_sliced[phase_map[index][0]][0]
                seg_2_lane_1 = num_in_deg_sliced[phase_map[index][0]][1]
                seg_3_lane_1 = num_in_deg_sliced[phase_map[index][0]][2] + num_in_deg_sliced[phase_map[index][0]][3]

                seg_1_lane_2 = num_in_deg_sliced[phase_map[index][1]][0]
                seg_2_lane_2 = num_in_deg_sliced[phase_map[index][1]][1]
                seg_3_lane_2 = num_in_deg_sliced[phase_map[index][1]][2] + num_in_deg_sliced[phase_map[index][1]][3]

                only_phase_state_text = (
                    "The signal chosen in previous step:\n"
                    f"Signal: {p}\n"
                    f"Allowed lanes: {phase_explanation_dict_detail[p][8:-1]}\n"
                    f"- Early queued: {queue_len_1} ({location_dict[lane_1[0]]}), {queue_len_2} ({location_dict[lane_2[0]]}), {queue_len_1 + queue_len_2} (Total)\n"
                    f"- Segment 1: {seg_1_lane_1} ({location_dict[lane_1[0]]}), {seg_1_lane_2} ({location_dict[lane_2[0]]}), {seg_1_lane_1 + seg_1_lane_2} (Total)\n"
                    f"- Segment 2: {seg_2_lane_1} ({location_dict[lane_1[0]]}), {seg_2_lane_2} ({location_dict[lane_2[0]]}), {seg_2_lane_1 + seg_2_lane_2} (Total)\n"
                    f"- Segment 3: {seg_3_lane_1} ({location_dict[lane_1[0]]}), {seg_3_lane_2} ({location_dict[lane_2[0]]}), {seg_3_lane_1 + seg_3_lane_2} (Total)\n\n"
                    f"The duration chosen in previous step:\n"
                    f"Duration : {history[1]}\n"
                    f"The passed vehicle number in this signal and dutation is {history[3]}\n"
                )
                total_text += only_phase_state_text

            if index == history[0] and i == 1:
                lane_1 = p[:2]
                lane_2 = p[2:]

                queue_len_1 = int(lane_queue_vehicle_in[phase_map[index][0]])
                queue_len_2 = int(lane_queue_vehicle_in[phase_map[index][1]])

                seg_1_lane_1 = num_in_deg_sliced[phase_map[index][0]][0]
                seg_2_lane_1 = num_in_deg_sliced[phase_map[index][0]][1]
                seg_3_lane_1 = num_in_deg_sliced[phase_map[index][0]][2] + num_in_deg_sliced[phase_map[index][0]][3]

                seg_1_lane_2 = num_in_deg_sliced[phase_map[index][1]][0]
                seg_2_lane_2 = num_in_deg_sliced[phase_map[index][1]][1]
                seg_3_lane_2 = num_in_deg_sliced[phase_map[index][1]][2] + num_in_deg_sliced[phase_map[index][1]][3]

                only_phase_state_text = (
                    "The signal chosen in two steps back:\n"
                    f"Signal: {p}\n"
                    f"Allowed lanes: {phase_explanation_dict_detail[p][8:-1]}\n"
                    f"- Early queued: {queue_len_1} ({location_dict[lane_1[0]]}), {queue_len_2} ({location_dict[lane_2[0]]}), {queue_len_1 + queue_len_2} (Total)\n"
                    f"- Segment 1: {seg_1_lane_1} ({location_dict[lane_1[0]]}), {seg_1_lane_2} ({location_dict[lane_2[0]]}), {seg_1_lane_1 + seg_1_lane_2} (Total)\n"
                    f"- Segment 2: {seg_2_lane_1} ({location_dict[lane_1[0]]}), {seg_2_lane_2} ({location_dict[lane_2[0]]}), {seg_2_lane_1 + seg_2_lane_2} (Total)\n"
                    f"- Segment 3: {seg_3_lane_1} ({location_dict[lane_1[0]]}), {seg_3_lane_2} ({location_dict[lane_2[0]]}), {seg_3_lane_1 + seg_3_lane_2} (Total)\n\n"
                    f"The duration chosen in two steps back:\n"
                    f"Duration : {history[1]}\n"
                    f"The passed vehicle number in this signal and dutation is {history[3]}\n"
                )
                total_text += only_phase_state_text
            if index == history[0] and i == 2:
                lane_1 = p[:2]
                lane_2 = p[2:]

                queue_len_1 = int(lane_queue_vehicle_in[phase_map[index][0]])
                queue_len_2 = int(lane_queue_vehicle_in[phase_map[index][1]])

                seg_1_lane_1 = num_in_deg_sliced[phase_map[index][0]][0]
                seg_2_lane_1 = num_in_deg_sliced[phase_map[index][0]][1]
                seg_3_lane_1 = num_in_deg_sliced[phase_map[index][0]][2] + num_in_deg_sliced[phase_map[index][0]][3]

                seg_1_lane_2 = num_in_deg_sliced[phase_map[index][1]][0]
                seg_2_lane_2 = num_in_deg_sliced[phase_map[index][1]][1]
                seg_3_lane_2 = num_in_deg_sliced[phase_map[index][1]][2] + num_in_deg_sliced[phase_map[index][1]][3]

                only_phase_state_text = (
                    "The signal chosen in three steps back:\n"
                    f"Signal: {p}\n"
                    f"Allowed lanes: {phase_explanation_dict_detail[p][8:-1]}\n"
                    f"- Early queued: {queue_len_1} ({location_dict[lane_1[0]]}), {queue_len_2} ({location_dict[lane_2[0]]}), {queue_len_1 + queue_len_2} (Total)\n"
                    f"- Segment 1: {seg_1_lane_1} ({location_dict[lane_1[0]]}), {seg_1_lane_2} ({location_dict[lane_2[0]]}), {seg_1_lane_1 + seg_1_lane_2} (Total)\n"
                    f"- Segment 2: {seg_2_lane_1} ({location_dict[lane_1[0]]}), {seg_2_lane_2} ({location_dict[lane_2[0]]}), {seg_2_lane_1 + seg_2_lane_2} (Total)\n"
                    f"- Segment 3: {seg_3_lane_1} ({location_dict[lane_1[0]]}), {seg_3_lane_2} ({location_dict[lane_2[0]]}), {seg_3_lane_1 + seg_3_lane_2} (Total)\n\n"
                    f"The duration chosen in three steps back:\n"
                    f"Duration : {history[1]}\n"
                    f"The passed vehicle number in this signal and dutation is {history[3]}\n"
                )
                total_text += only_phase_state_text
        return total_text



    def action2code(self, signal_text_phase, signal_text_duration):
            phase =  self.phases[signal_text_phase]
            duration = signal_text_duration
            return phase, duration

    def one_text(self, i, history, lane_queue_vehicle_in,num_in_deg_sliced, phase_map, one_piece_history_for_expert, one_piece_history_for_expert_num):

        for p, index in self.phases.items():
            if index == history[0] and i == 0:
                lane_1 = p[:2]
                lane_2 = p[2:]

                queue_len_1 = int(lane_queue_vehicle_in[phase_map[index][0]])
                queue_len_2 = int(lane_queue_vehicle_in[phase_map[index][1]])

                seg_1_lane_1 = num_in_deg_sliced[phase_map[index][0]][0]
                seg_2_lane_1 = num_in_deg_sliced[phase_map[index][0]][1]
                seg_3_lane_1 = num_in_deg_sliced[phase_map[index][0]][2] + num_in_deg_sliced[phase_map[index][0]][3]

                seg_1_lane_2 = num_in_deg_sliced[phase_map[index][1]][0]
                seg_2_lane_2 = num_in_deg_sliced[phase_map[index][1]][1]
                seg_3_lane_2 = num_in_deg_sliced[phase_map[index][1]][2] + num_in_deg_sliced[phase_map[index][1]][3]

                one_piece_history_for_expert = (
                    f"Signal: {p}\n"
                    f"Allowed lanes: {phase_explanation_dict_detail[p][8:-1]}\n"
                    f"- Early queued: {queue_len_1} ({location_dict[lane_1[0]]}), {queue_len_2} ({location_dict[lane_2[0]]}), {queue_len_1 + queue_len_2} (Total)\n"
                    f"- Segment 1: {seg_1_lane_1} ({location_dict[lane_1[0]]}), {seg_1_lane_2} ({location_dict[lane_2[0]]}), {seg_1_lane_1 + seg_1_lane_2} (Total)\n"
                    f"- Segment 2: {seg_2_lane_1} ({location_dict[lane_1[0]]}), {seg_2_lane_2} ({location_dict[lane_2[0]]}), {seg_2_lane_1 + seg_2_lane_2} (Total)\n"
                    f"- Segment 3: {seg_3_lane_1} ({location_dict[lane_1[0]]}), {seg_3_lane_2} ({location_dict[lane_2[0]]}), {seg_3_lane_1 + seg_3_lane_2} (Total)\n\n"
                    f"The duration chosen in previous step:\n"
                    f"Duration : {history[1]}\n"
                    f"The passed vehicle number in this signal and dutation is {history[3]}\n"
                )
                one_piece_history_for_expert_num = [queue_len_1 + queue_len_2 ,seg_1_lane_1 + seg_1_lane_2, seg_2_lane_1 + seg_2_lane_2, seg_3_lane_1 + seg_3_lane_2,history[1], history[3]]

        return one_piece_history_for_expert, one_piece_history_for_expert_num

    def append_to_json_file(self,file_path, data):
        file_exists = os.path.exists(file_path)
        if not file_exists:
            with open(file_path, 'w') as file:
                file.write('[\n')
        with open(file_path, 'r+') as file:
            file.seek(0, os.SEEK_END)
            file_size = file.tell()

            if file_size > 2:
                file.seek(file_size - 2)
                file.truncate()
                file.write(',\n')
            json.dump(data, file, indent=None)
            file.write('\n]')

    def heraldmaker_v2(self, vehicles_per_lane, waiting_num, speed, distance, lane_length):


        distance_all_dict = {}
        for lane, vehicle_list in vehicles_per_lane.items():
            distance_dict = {}
            for vehicle in vehicle_list:
                if 'shadow' not in vehicle:
                    distance_dict[vehicle] = distance[vehicle]
            sorted_dict = sorted(distance_dict.items(), key=lambda x: x[1], reverse=True)
            distance_all_dict[lane] = sorted_dict

        stop_points = {}
        for lane in distance_all_dict:
            stop_points[lane] = lane_length[lane] - 16

        distance_all_dict_running = {}
        for lane in distance_all_dict:
            distance_all_dict_running[lane] = distance_all_dict[lane][waiting_num[lane]:]

        total_temp_t = 0
        times_ceiling = 0
        remember_the_split = []
        waiting_num_archive = []

        min_durations = {}
        for lane in distance_all_dict_running:
            if distance_all_dict_running[lane] != []:
                min_durations[lane] = []
                for index, vehicle in enumerate(distance_all_dict_running[lane]):
                    min_durations[lane].append(
                        math.ceil((stop_points[lane] - distance_all_dict_running[lane][index][1]) / 11))
                    # (11 if speed[distance_all_dict_running[lane][index][0]] == 0 else (speed[distance_all_dict_running[lane][index][0]]))))
        sorted_min_durations = sorted(min_durations.items(), key=lambda item: item[1])

        updated_waiting = {}
        if sorted_min_durations != []:

            updated_waiting = waiting_num.copy()

            for road, times in sorted_min_durations:
                current_waiting = updated_waiting.get(road, 0)

                duration = max(current_waiting * 3 + 3, 5)


                for arrival_time in sorted(times):
                    if arrival_time <= duration:
                        current_waiting += 1
                        duration = current_waiting * 3 + 3
                        print(
                            f"    车辆可在持续时间内通过。更新后的等待数量: {current_waiting}, 新的持续时间: {duration}")
                    else:
                        print(f"    车辆无法在持续时间内通过。等待数量保持不变。")

                updated_waiting[road] = current_waiting

        return updated_waiting

    def calculate_phase_queue_sums_v2(self, waiting_num_archive):
        phase_queue_sums = []
        phase_queue_origins = []
        if waiting_num_archive != {}:
            phase_sum = []
            phase_origin = []
            for phase, lanes_controlled in self.dic_traffic_env_conf["PHASETOTAL"].items():
                queued_num_all_lanes = list(waiting_num_archive.values())
                temp_sum_variable = 0
                temp_origin_variable = []
                for index, control in enumerate(lanes_controlled):
                    if control == 1 and index not in [2,5,8,11]:
                        temp_sum_variable += queued_num_all_lanes[index]
                        if queued_num_all_lanes[index] > 0:
                            temp_origin_variable.append(queued_num_all_lanes[index])
                phase_sum.append(temp_sum_variable)
                phase_origin.append(temp_origin_variable)

            phase_queue_sums.append(phase_sum)
            phase_queue_origins.append(phase_origin)

        return phase_queue_sums, phase_queue_origins

    def choose_max_sublist(self, arrays):

        if len(arrays) == 5:
            arrays = arrays[:4]
        results = []

        for i, sublist in enumerate(arrays):
            if not sublist:
                sum_val = 0
                diff = 0
            elif len(sublist) == 1:
                sum_val = sublist[0]
                diff = 0
            else:
                sum_val = sum(sublist)
                diff = max(sublist) - min(sublist)

            results.append((i, sublist, sum_val, diff))

        max_sum = max(r[2] for r in results)

        candidates = [r for r in results if r[2] == max_sum]

        if len(candidates) == 1:
            return candidates[0][0], candidates[0][1]

        min_diff = min(c[3] for c in candidates)
        finalists = [c for c in candidates if c[3] == min_diff]

        if len(finalists) == 1:
            return finalists[0][0], finalists[0][1]


        choice = random.choice(finalists)
        return choice[0], choice[1]



