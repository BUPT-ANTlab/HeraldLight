"""
Change the function of testexp: test the transfer ability on other datasets
Step: 1. pretain the model
      2. preapre the new data
      3. test the model on the new data
Liang Zhang
"""
import json
import os
import time
from multiprocessing import Process
from utils import config
from utils.utils import merge
from utils.cityflow_env import CityFlowEnv
import argparse
import shutil
from collections import deque
import numpy as np
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


multi_process = True


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-memo", type=str, default='benchmark_1216_c_1')
    parser.add_argument("-old_memo", type=str, default='benchmark_0107_1')
    parser.add_argument("-model", type=str, default="dynamiclight")
    parser.add_argument("-old_dir", type=str, default='anon_3_4_jinan_real.json_10_31_15_59_21')
    parser.add_argument("-old_round", type=int, default=200)

    parser.add_argument("-workers", type=int, default=3)

    parser.add_argument("-net1", action="store_true", default=0)
    parser.add_argument("-net2", action="store_true", default=0)
    parser.add_argument("-net3", action="store_true", default=0)

    parser.add_argument("-hangzhou", action="store_true", default=0)
    parser.add_argument("-jinan", action="store_true", default=1)
    parser.add_argument("-newyork2", action="store_true", default=0)

    return parser.parse_args()


def main(args):
    if args.net1:
        count, count2 = 3600, 3600
        road_net, num_lanes, num_lane = "3_4", [2, 2, 2, 2], 8
        phase_map = [[1, 3], [5, 7], [0, 2], [4, 6]]
        traffic_file_list = ["anon_3_4_6320.json", "anon_3_4_4797.json"]
        num_rounds = 80
        template = "Network1"
    elif args.net2:
        count, count2 = 3600, 3600
        road_net, num_lanes, num_lane = "3_4", [3, 3, 2, 2], 10
        phase_map = [[1, 4], [7, 9], [0, 3], [6, 8]]
        traffic_file_list = ["anon_3_4_4805.json", "anon_3_4_6277.json"]
        num_rounds = 80
        template = "Network2"
    elif args.net3:
        count, count2 = 3600, 3600
        road_net, num_lanes, num_lane = "3_4", [4, 4, 4, 4], 16
        phase_map = [[1, 2, 5, 6], [9, 10, 13, 14], [0, 4], [8, 12]]
        traffic_file_list = ["anon_3_4_4785.json", "anon_3_4_6247.json"]
        num_rounds = 80
        template = "Network3"
    elif args.hangzhou:
        count, count2 = 3600, 3600
        road_net, num_lanes, num_lane = "4_4", [3, 3, 3, 3], 12
        phase_map = [[1, 4], [7, 10], [0, 3], [6, 9]]
        traffic_file_list = ["anon_4_4_hangzhou_real_5816.json"] #"anon_4_4_hangzhou_real_5816.json"
        num_rounds = 80
        template = "Hangzhou"
    elif args.jinan:
        count, count2 = 3600, 3600
        road_net, num_lanes, num_lane = "3_4", [3, 3, 3, 3], 12
        phase_map = [[1, 4], [7, 10], [0, 3], [6, 9]]
        traffic_file_list = ["anon_3_4_jinan_real.json"]
        num_rounds = 80
        template = "Jinan"

    elif args.newyork2:
        count, count2 = 3600, 3600
        road_net, num_lanes, num_lane = "28_7", [3, 3, 3, 3], 12
        phase_map = [[1, 4], [7, 10], [0, 3], [6, 9]]
        traffic_file_list = ["anon_28_7_newyork_real_triple.json"]
        num_rounds = 1
        template = "newyork_28_7"

    NUM_ROW = int(road_net.split('_')[0])
    NUM_COL = int(road_net.split('_')[1])
    num_intersections = NUM_ROW * NUM_COL
    print('num_intersections:', num_intersections)
    print(traffic_file_list)

    old_memo = args.old_memo
    old_dir = args.old_dir
    old_model_path = os.path.join("model", old_memo, old_dir)

    process_list = []
    n_workers = args.workers

    for traffic_file in traffic_file_list:
        dic_traffic_env_conf_extra = {
            "OLD_ROUND2": args.old_round,
            "NUM_LANES": num_lanes,
            "PHASE_MAP": phase_map,
            "NUM_LANE": num_lane,
            "NUM_ROUNDS": num_rounds,
            "NUM_ROUNDS": num_rounds,
            "NUM_GENERATORS": 1,
            "NUM_AGENTS": 1,
            "NUM_INTERSECTIONS": num_intersections,
            "RUN_COUNTS": count,
            "MODEL_NAME": args.model,
            "NUM_ROW": NUM_ROW,
            "NUM_COL": NUM_COL,
            "TRAFFIC_FILE": traffic_file,
            "ROADNET_FILE": "roadnet_{0}.json".format(road_net),
            "LIST_STATE_FEATURE": [
                "phase_total",
                "lane_queue_vehicle_in",
                "lane_run_in_part",
                "num_in_deg",
            ],
            "LIST_STATE_FEATURE_1": [
                "phase_total",
                "lane_queue_vehicle_in",
                "lane_run_in_part",
                "num_in_deg",

            ],
            "LIST_STATE_FEATURE_2": [
                "num_in_deg",
            ],

            "DIC_REWARD_INFO": {
                "queue_length": -0.25,
            },
        }

        if args.net1:
            dic_traffic_env_conf_extra["PHASE"] = {
                1: [0, 1, 0, 1, 0, 0, 0, 0],
                2: [0, 0, 0, 0, 0, 1, 0, 1],
                3: [1, 0, 1, 0, 0, 0, 0, 0],
                4: [0, 0, 0, 0, 1, 0, 1, 0]
            }
            dic_traffic_env_conf_extra["PHASETOTAL"] = {
                1: [0, 1, 0, 1, 0, 0, 0, 0],
                2: [0, 0, 0, 0, 0, 1, 0, 1],
                3: [1, 0, 1, 0, 0, 0, 0, 0],
                4: [0, 0, 0, 0, 1, 0, 1, 0]
            }
        elif args.net2:
            dic_traffic_env_conf_extra["PHASE"] = {
                1: [0, 1, 0, 1, 0, 0, 0, 0],
                2: [0, 0, 0, 0, 0, 1, 0, 1],
                3: [1, 0, 1, 0, 0, 0, 0, 0],
                4: [0, 0, 0, 0, 1, 0, 1, 0]
            }
            dic_traffic_env_conf_extra["PHASETOTAL"] = {
                1: [0, 1, 1, 0, 1, 1, 0, 0, 0, 0],
                2: [0, 0, 1, 0, 0, 1, 0, 1, 0, 1],
                3: [1, 0, 1, 1, 0, 1, 0, 0, 0, 0],
                4: [0, 0, 1, 0, 0, 1, 1, 0, 1, 0]
            }
        elif args.net3:
            dic_traffic_env_conf_extra["PHASE"] = {
                1: [0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0],
                2: [0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1],
                3: [1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                4: [0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0]
            }
            dic_traffic_env_conf_extra["PHASETOTAL"] = {
                1: [0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1],
                2: [0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1],
                3: [1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1],
                4: [0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1]
            }
        else:
            dic_traffic_env_conf_extra["PHASE"] = {
                1: [0, 1, 0, 1, 0, 0, 0, 0],
                2: [0, 0, 0, 0, 0, 1, 0, 1],
                3: [1, 0, 1, 0, 0, 0, 0, 0],
                4: [0, 0, 0, 0, 1, 0, 1, 0]
            }
            dic_traffic_env_conf_extra["PHASETOTAL"] = {
                1: [0, 1, 1, 0, 1, 1, 0, 0, 1, 0, 0, 1],
                2: [0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 1],
                3: [1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 0, 1],
                4: [0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 0, 1]
            }
        #这里发现了优化5秒间隔的优化算法之后尝试构造数据集
        dic_traffic_env_conf_extra["ACTION_DURATION_gpt"] = {
            '10': 10, '11': 11, '12': 12, '13': 13, '14': 14, '15': 15,
            '16': 16, '17': 17, '18': 18, '19': 19, '20': 20, '21': 21,
            '22': 22, '23': 23, '24': 24, '25': 25, '26': 26, '27': 27,
            '28': 28, '29': 29, '30': 30, '31': 31, '32': 32, '33': 33,
            '34': 34, '35': 35, '36': 36, '37': 37, '38': 38, '39': 39,
            '40': 40
        }
        # change the model path to the old model path
        dic_path = {
            "PATH_TO_MODEL": old_model_path,  # use old model path
            "PATH_TO_WORK_DIRECTORY": os.path.join("records", args.memo, traffic_file + "_" +
                                                   time.strftime('%m_%d_%H_%M_%S', time.localtime(
                                                       time.time()))),
            "PATH_TO_DATA": os.path.join("data", template, str(road_net))
        }
        deploy_dic_agent_conf = getattr(config, "DIC_BASE_AGENT_CONF")
        deploy_dic_traffic_env_conf = merge(config.dic_traffic_env_conf, dic_traffic_env_conf_extra)
        # old configs
        old_confs = []

        if multi_process:
            tsr = Process(target=testor_wrapper,
                          args=(deploy_dic_agent_conf,
                                deploy_dic_traffic_env_conf,
                                dic_path,
                                old_confs))
            process_list.append(tsr)
        else:
            testor_wrapper(deploy_dic_agent_conf,
                           deploy_dic_traffic_env_conf,
                           dic_path,
                           old_confs)

    if multi_process:
        for i in range(0, len(process_list), n_workers):
            i_max = min(len(process_list), i + n_workers)
            for j in range(i, i_max):
                print(j)
                print("start_traffic")
                process_list[j].start()
                print("after_traffic")
            for k in range(i, i_max):
                print("traffic to join", k)
                process_list[k].join()
                print("traffic finish join", k)

    return args.memo


def testor_wrapper(dic_agent_conf, dic_traffic_env_conf, dic_path, old_confs):
    testor = Testor(dic_agent_conf,
                    dic_traffic_env_conf,
                    dic_path,
                    old_confs)
    testor.main()
    print("============= restor wrapper end =========")


class Testor:
    def __init__(self, dic_agent_conf, dic_traffic_env_conf, dic_path, old_confs):
        self.dic_agent_conf = dic_agent_conf
        self.dic_traffic_env_conf = dic_traffic_env_conf
        self.dic_path = dic_path

        self.dic_agent_conf["EPSILON"] = 0
        self.dic_agent_conf["MIN_EPSILON"] = 0

        self._path_check()
        self._copy_conf_file()
        self._copy_anon_file()
        agent_name = self.dic_traffic_env_conf["MODEL_NAME"]
        # use one-model
        self.agent = config.DIC_AGENTS[agent_name](
            dic_agent_conf=dic_agent_conf,
            dic_traffic_env_conf=dic_traffic_env_conf,
            dic_path=dic_path,
            cnt_round=99999,
            intersection_id=str(0)
        )

        self.path_to_log = self.dic_path["PATH_TO_WORK_DIRECTORY"]
        if not os.path.exists(self.path_to_log):
            os.makedirs(self.path_to_log)

        self.four_states_prompt = []
        self.phases = four_phase_list
        self.log_file_path = ""
        self.first_time = True

    def main(self):
        rounds = ["round_" + str(i) for i in range(self.dic_traffic_env_conf["OLD_ROUND2"] - 10,
                                                   self.dic_traffic_env_conf["OLD_ROUND2"])]
        cnt_rounds = [i for i in range(self.dic_traffic_env_conf["OLD_ROUND2"] - 10,
                                       self.dic_traffic_env_conf["OLD_ROUND2"])]

        #只跑一次
        one = True

        for i, old_round in enumerate(rounds):
            if one:
                self.path_to_log = os.path.join(self.dic_path["PATH_TO_WORK_DIRECTORY"], "test_round", old_round)
                if not os.path.exists(self.path_to_log):
                    os.makedirs(self.path_to_log)
                self.env = CityFlowEnv(path_to_log=self.path_to_log,
                                       path_to_work_directory=self.dic_path["PATH_TO_WORK_DIRECTORY"],
                                       dic_traffic_env_conf=self.dic_traffic_env_conf)
                #self.agent.load_network("{0}_inter_0".format(old_round))

                self.agent.cnt_round = cnt_rounds[i]
                self.agent.load_network(file_name="{0}_inter_0".format('round_199'),
                                        file_path=r'records/benchmark_0107_1/hangzhou1')
                self.run()
                one = False

    def run(self):
        state, step_time, list_need = self.env.reset()
        running_start_time = time.time()

        vehicle_pass_num = {}
        # 初始化 vehicle_pass_num 字典
        vehicle_pass_num = {inter: 0 for inter in range(self.env.num_intersection)}
        history_intersection = {inter: [] for inter in range(self.env.num_intersection)}
        self.env.list_gpt_history = [deque(maxlen=3) for _ in range(self.env.num_intersection)]
        one_time_toggle = False

        queue_length_episode = []
        waiting_time_episode = []

        self.log_file_path = os.path.join(self.path_to_log, 'synthetic_prompts1.json')


        if self.first_time:
            with open(self.log_file_path, 'w') as f:
                f.write("[\n")
            self.first_time = False

        while step_time < self.dic_traffic_env_conf["RUN_COUNTS"]:
            step_start_time = time.time()
            phase_action, duration_action = self.agent.choose_action(state, list_need)

            #syn_prompts = self.synthetic(state, list_need, self.env.list_intersection, self.env.list_gpt_history)
            syn_prompts = ""
            results = [
                {
                    "step_time": step_time,
                    "syn_prompts": prompt,
                    "phase_action": four_phase_list_reverse[phase],
                    "duration_action": int(self.dic_traffic_env_conf["ACTION_DURATION"][duration]),
                }
                for prompt, phase, duration in zip(syn_prompts, phase_action, duration_action)
            ]

            with open(self.log_file_path, 'a') as f:
                f.write(json.dumps(results, ensure_ascii=False, indent=4)[1:-1])
                if step_time < self.dic_traffic_env_conf["RUN_COUNTS"] - 1:
                    f.write(",\n")



            log_dir = self.path_to_log
            log_filename = "step_info_index_5.txt"
            log_file_path = os.path.join(log_dir, log_filename)
            if 5 in list_need:
                indices = [i for i, x in enumerate(list_need) if x == 5]
                with open(log_file_path, 'a') as f:
                    for index in indices:
                        phase = phase_action[index]
                        duration = duration_action[index]
                        f.write(f"{int(step_time)}: 相位：{four_phase_list_reverse[phase]}, 持续时间：{duration}\n")

            next_state, step_time, list_need = self.env.step_gpt(phase_action, duration_action, vehicle_pass_num,
                                                                 history_intersection)

            print("time: {0}, running_time: {1}".format(self.env.get_current_time(), time.time() - step_start_time))

            queue_length_inter = [sum(inter.dic_feature['lane_queue_vehicle_in']) for inter in
                                  self.env.list_intersection]
            queue_length_episode.append(sum(queue_length_inter))

            waiting_times = [self.env.waiting_vehicle_list[veh]['time'] for veh in self.env.waiting_vehicle_list]
            waiting_time_episode.append(np.mean(waiting_times) if waiting_times else 0.0)

            state = next_state

        with open(self.log_file_path, 'a') as f:
            f.write("\n]")

        running_time = time.time() - running_start_time
        log_start_time = time.time()
        print("=========== start env logging ===========")
        self.env.batch_log_2()
        log_time = time.time() - log_start_time
        # self.env.end_anon()
        print("running_time: ", running_time)
        print("log_time: ", log_time)

        #################################################
        vehicle_travel_times = {}
        for inter in self.env.list_intersection:
            arrive_left_times = inter.dic_vehicle_arrive_leave_time
            for veh in arrive_left_times:
                if "shadow" in veh:
                    continue
                enter_time = arrive_left_times[veh]["enter_time"]
                leave_time = arrive_left_times[veh]["leave_time"]
                if not np.isnan(enter_time):
                    leave_time = leave_time if not np.isnan(leave_time) else self.dic_traffic_env_conf["RUN_COUNTS"]
                    if veh not in vehicle_travel_times:
                        vehicle_travel_times[veh] = [leave_time - enter_time]
                    else:
                        vehicle_travel_times[veh].append(leave_time - enter_time)

        total_travel_time = np.mean([sum(vehicle_travel_times[veh]) for veh in vehicle_travel_times])
        # print('total_travel_time\n\n\n',total_travel_time)

        results = {
            "training_avg_queue_len": np.mean(queue_length_episode) if len(queue_length_episode) > 0 else 0,
            "training_avg_travel_time": total_travel_time,
            "training_avg_waiting_time": np.mean(waiting_time_episode) if len(queue_length_episode) > 0 else 0,
        }
        print(results)

    def _path_check(self):
        # check path
        if os.path.exists(self.dic_path["PATH_TO_WORK_DIRECTORY"]):
            if self.dic_path["PATH_TO_WORK_DIRECTORY"] != "records/default":
                raise FileExistsError
            else:
                pass
        else:
            os.makedirs(self.dic_path["PATH_TO_WORK_DIRECTORY"])

    def _copy_conf_file(self, path=None):
        if path is None:
            path = self.dic_path["PATH_TO_WORK_DIRECTORY"]
        json.dump(self.dic_agent_conf, open(os.path.join(path, "agent.conf"), "w"),
                  indent=4)
        json.dump(self.dic_traffic_env_conf,
                  open(os.path.join(path, "traffic_env.conf"), "w"), indent=4)

    def _copy_anon_file(self, path=None):
        if path is None:
            path = self.dic_path["PATH_TO_WORK_DIRECTORY"]
        shutil.copy(os.path.join(self.dic_path["PATH_TO_DATA"], self.dic_traffic_env_conf["TRAFFIC_FILE"]),
                    os.path.join(path, self.dic_traffic_env_conf["TRAFFIC_FILE"]))
        shutil.copy(os.path.join(self.dic_path["PATH_TO_DATA"], self.dic_traffic_env_conf["ROADNET_FILE"]),
                    os.path.join(path, self.dic_traffic_env_conf["ROADNET_FILE"]))


    def synthetic(self,states, list_need, list_intersection, list_gpt_history):
        pactions = []
        dactions = []
        prompts = []
        self.four_states_prompt = []

        for i, intersection_index in enumerate(list_need):
            state = states[i]
            state_text, state_text_dict = self.state_to_text(state)

            self.four_states_prompt.append(state_text_dict)

            relieve_history, one_piece_history_for_expert = self.relieve_waiting_vehicles_history(intersection_index,
                                                                                                  list_gpt_history)
            if relieve_history == None:
                relieve_history = ""

            prompts.append(self.get_prompt(state_text, relieve_history))



        return prompts



    def relieve_waiting_vehicles_history(self, intersection_index, list_gpt_history):
        return self.state_to_text_for_history(list_gpt_history[intersection_index])
        #print('list_need, list_gpt_history',intersection_index, list_gpt_history[intersection_index])


if __name__ == "__main__":
    args = parse_args()
    main(args)
