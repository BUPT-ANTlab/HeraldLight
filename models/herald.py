from .agent import Agent
import random
from collections import defaultdict
import numpy as np


class heraldagent(Agent):

    def __init__(self, dic_agent_conf, dic_traffic_env_conf, dic_path, cnt_round, intersection_id):

        super(heraldagent, self).__init__(dic_agent_conf, dic_traffic_env_conf, dic_path, intersection_id)

        self.current_phase_time = 0
        self.phase_length = len(self.dic_traffic_env_conf["PHASE"])
        self.last_step_list_need = []
        self.last_queue_num = [0 for i in range(self.dic_traffic_env_conf['NUM_INTERSECTIONS'])]
        self.action = None
        self.first_time = True
        self.flow_length_stats = defaultdict(lambda: defaultdict(int))
        self.queue2duration = {
            0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5,
            6: 6, 7: 7, 8: 8, 9: 9,
            10: 10, 11: 11, 12: 12, 13: 13, 14: 14, 15: 15,
            16: 16, 17: 17, 18: 18, 19: 19, 20: 20, 21: 21,
            22: 22, 23: 23, 24: 24, 25: 25, 26: 26, 27: 27,
            28: 28, 29: 29, 30: 30, 31: 31, 32: 32, 33: 33,
            34: 34, 35: 35, 36: 36, 37: 37, 38: 38, 39: 39,
            40: 40
        }

        if self.phase_length == 4:
            self.DIC_PHASE_MAP_4 = {  # for 4 phase
                1: 0,
                2: 1,
                3: 2,
                4: 3,
                0: 0
            }
        elif self.phase_length == 8:
            self.DIC_PHASE_MAP = {
                1: 0,
                2: 1,
                3: 2,
                4: 3,
                5: 4,
                6: 5,
                7: 6,
                8: 7,
                0: 0
            }

    def choose_action(self, states, list_need, list_intersection,action_static , vehicle_pass_num_duration_left,step_time):

        pactions = []
        dactions = []


        for i, intersection_index in enumerate(list_need):
            # dynamic_plug_in_choice = four_phase_list_reverse[phase_action_plug_in[i]]

            four_states_prompt = []
            state = states[i]
            phase_sums = {}
            phase_early_queued = {}
            queued_time_dict = list_intersection[intersection_index].dic_vehicle_queued_time_in

            # herald计算
            current_vehicles_per_lane = list_intersection[intersection_index].dic_lane_vehicle_current_step
            current_waiting_vechicles_num_per_lane_in = list_intersection[
                intersection_index].dic_lane_waiting_vehicle_count_current_step
            current_vehicles_speed_per_vehicle = list_intersection[intersection_index].dic_vehicle_speed_current_step
            current_vehicles_distance_per_vehicle = list_intersection[
                intersection_index].dic_vehicle_distance_current_step
            lane_length = list_intersection[intersection_index].lane_length

            queue_snap_no_herald, queue_snap_origin_no_herald = self.calculate_phase_queue_sums_v2(
                current_waiting_vechicles_num_per_lane_in)

            if queue_snap_origin_no_herald != []:
                max_a, max_b = self.choose_max_sublist(queue_snap_origin_no_herald[0])
                print("max_a, max_b", max_a, max_b)

            if queue_snap_origin_no_herald != [] and max_b != []:
                paction = max_a
                print("self.last_step_list_need: ",self.last_step_list_need,"\n","last_queue_num ",self.last_queue_num,"\n","list_need: ",list_need, "\n", "action_static: ", action_static, "\n", "vehicle_pass_num_duration_left: ", vehicle_pass_num_duration_left)
                self.last_queue_num[intersection_index] = max(max_b)

                if self.last_queue_num != []:
                    for intersection_index in list_need:
                        if self.last_queue_num[intersection_index] > len(vehicle_pass_num_duration_left[intersection_index]):


                            self.queue2duration[self.last_queue_num[intersection_index]] += 1

                        elif self.last_queue_num[intersection_index] < len(vehicle_pass_num_duration_left[intersection_index]):
                            target_rest = list(vehicle_pass_num_duration_left[intersection_index][
                                     -self.last_queue_num[intersection_index]].values())[0]
                            self.queue2duration[self.last_queue_num[intersection_index]] -= 1

                        else:
                            if vehicle_pass_num_duration_left[intersection_index] != []:
                                self.queue2duration[self.last_queue_num[intersection_index]] -= list(vehicle_pass_num_duration_left[intersection_index][0].values())[0] -1 if list(vehicle_pass_num_duration_left[intersection_index][0].values())[0]>1 else 0

                for i in range(len(self.queue2duration)):
                    if self.queue2duration[i] < 0:
                        self.queue2duration[i] = 0

                daction = self.queue2duration[max(max_b)]


                if daction > 40:
                    daction = 40
                print(f"{paction, daction}")
            else:
                paction = random.randint(0, 3)
                daction = 5
                print(f"{paction, daction}")

            pactions.append(paction)
            dactions.append(daction)

        self.last_step_list_need = list_need
        print(self.queue2duration)

        self.queue2duration_new = {
            key: max(value_list, key=lambda x: x["count"])["duration"]
            for key, value_list in self.format_stats(self.flow_length_stats).items()
        }
        print("↓" * 10)
        print("Method 1", self.queue2duration)
        print("Method 2", self.format_stats(self.flow_length_stats))
        print("Method 2.1", self.queue2duration_new)
        print("↑" * 10)

        return pactions, dactions


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
                    if control == 1 and index not in [2, 5, 8, 11]:
                        temp_sum_variable += queued_num_all_lanes[index]
                        if queued_num_all_lanes[index] > 0:
                            temp_origin_variable.append(queued_num_all_lanes[index])
                phase_sum.append(temp_sum_variable)
                phase_origin.append(temp_origin_variable)

            phase_queue_sums.append(phase_sum)
            phase_queue_origins.append(phase_origin)

        return phase_queue_sums, phase_queue_origins
    def format_stats(self, raw_stats):
        formatted = {}
        for pos, time_counts in raw_stats.items():
            formatted[pos] = [
                {"duration": time, "count": count}
                for time, count in time_counts.items()
            ]
        return formatted

    def accumulate_flow_stats(self, data, stats):
        for entry in data.values():
            if entry != []:
                for road in entry.values():
                    for position, flow in enumerate(road, start=1):
                        time = list(flow.values())[0]
                        stats[position][time] += 1

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


