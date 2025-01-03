from utils.utils import pipeline_wrapper, merge
from utils import config
import time
from multiprocessing import Process
import argparse
import os


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-memo", type=str, default='benchmark_0107_1')
    parser.add_argument("-mod", type=str, default="llama3170B")
    parser.add_argument("-gen", type=int, default=1)
    parser.add_argument("-multi_process", action="store_true", default=0)
    parser.add_argument("-workers", type=int, default=2)
    parser.add_argument("-net1", action="store_true", default=0)
    parser.add_argument("-net2", action="store_true", default=0)
    parser.add_argument("-net3", action="store_true", default=0)
    parser.add_argument("-hangzhou", action="store_true", default=0)
    parser.add_argument("-jinan", action="store_true", default=1)
    parser.add_argument("-newyork2", action="store_true", default=0)
    parser.add_argument("--gpt_version", type=str, default="llama3170B")
    parser.add_argument("--prompt", type=str, default='Commonsense')
    parser.add_argument("--wandb_name", type=str,
                        default='default_name')

    return parser.parse_args()


def main(in_args=None):
    if in_args.net1:
        count, count2 = 3600, 3600
        road_net, num_lanes, num_lane = "3_4", [2, 2, 2, 2], 8
        phase_map = [[1, 3], [5, 7], [0, 2], [4, 6]]
        traffic_file_list = ["anon_3_4_6320.json", "anon_3_4_4797.json"]
        num_rounds = 200
        template = "Network1"
    elif in_args.net2:
        count, count2 = 3600, 3600
        road_net, num_lanes, num_lane = "3_4", [3, 3, 2, 2], 10
        phase_map = [[1, 4], [7, 9], [0, 3], [6, 8]]
        traffic_file_list = ["anon_3_4_4805.json", "anon_3_4_6277.json"]
        num_rounds = 200
        template = "Network2"
    elif in_args.net3:
        count, count2 = 3600, 3600
        road_net, num_lanes, num_lane = "3_4", [4, 4, 4, 4], 16
        phase_map = [[1, 2, 5, 6], [9, 10, 13, 14], [0, 4], [8, 12]]
        traffic_file_list = ["anon_3_4_4785.json", "anon_3_4_6247.json"]
        num_rounds = 160
        template = "Network3"
    elif in_args.hangzhou:
        count, count2 = 3600, 3600
        road_net, num_lanes, num_lane = "4_4", [3, 3, 3, 3], 12
        phase_map = [[1, 4], [7, 10], [0, 3], [6, 9]]
        traffic_file_list = ["anon_4_4_hangzhou_real.json", "anon_4_4_hangzhou_real_5816.json"]
        num_rounds = 200
        template = "Hangzhou"
    elif in_args.jinan:
        count, count2 = 3600, 3600
        road_net, num_lanes, num_lane = "3_4", [3, 3, 3, 3], 12
        phase_map = [[1, 4], [7, 10], [0, 3], [6, 9]]
        traffic_file_list = ["anon_3_4_jinan_real.json"]
        ########################################################3333
        num_rounds = 1
        template = "Jinan"
    elif args.newyork2:
        count, count2 = 3600, 3600
        road_net, num_lanes, num_lane = "28_7", [3, 3, 3, 3], 12
        phase_map = [[1, 4], [7, 10], [0, 3], [6, 9]]
        traffic_file_list = ["anon_28_7_newyork_real_triple.json", "anon_28_7_newyork_real_double.json"]
        num_rounds = 80
        template = "newyork_28_7"

    if in_args.prompt == "Commonsense":
        in_args.memo = "ChatGPTTLCSCommonsense"

    NUM_COL = int(road_net.split('_')[1])
    NUM_ROW = int(road_net.split('_')[0])
    num_intersections = NUM_ROW * NUM_COL
    print('num_intersections:', num_intersections)
    print(traffic_file_list)

    process_list = []
    for traffic_file in traffic_file_list:

        dic_agent_conf_extra = {
            "GPT_VERSION": in_args.gpt_version,
            "LOG_DIR": "./GPT_logs",
        }

        dic_traffic_env_conf_extra = {
            "OBS_LENGTH": 167,
            "NUM_LANES": num_lanes,
            "PHASE_MAP": phase_map,
            "NUM_LANE": num_lane,
            "NUM_ROUNDS": num_rounds,

            "NUM_GENERATORS": in_args.gen,
            "NUM_AGENTS": 1,
            "NUM_INTERSECTIONS": num_intersections,
            "RUN_COUNTS": count,
            "RUN_COUNTS2": count2,
            "MODEL_NAME": in_args.mod,
            "WANDB_NAME": in_args.wandb_name,
            "NUM_ROW": NUM_ROW,
            "NUM_COL": NUM_COL,
            "TRAFFIC_FILE": traffic_file,
            "ROADNET_FILE": "roadnet_{0}.json".format(road_net),
            "TRAFFIC_SEPARATE": traffic_file,
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
        dic_path_extra = {
            "PATH_TO_MODEL": os.path.join("model", in_args.memo, traffic_file + "_"
                                          + time.strftime('%m_%d_%H_%M_%S', time.localtime(time.time()))),
            "PATH_TO_WORK_DIRECTORY": os.path.join("records", in_args.memo, traffic_file + "_"
                                                   + time.strftime('%m_%d_%H_%M_%S', time.localtime(time.time()))),
            "PATH_TO_DATA": os.path.join("data", template, str(road_net)),
            "PATH_TO_ERROR": os.path.join("errors", in_args.memo)
        }

        if in_args.net1:
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
        elif in_args.net2:
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
        elif in_args.net3:
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
                4: [0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 0, 1],
                999:[0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1]
            }
            dic_traffic_env_conf_extra["ACTION_DURATION_gpt"] = {
                '0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5,
                '6': 6, '7': 7, '8': 8, '9': 9,
                '10': 10, '11': 11, '12': 12, '13': 13, '14': 14, '15': 15,
                '16': 16, '17': 17, '18': 18, '19': 19, '20': 20, '21': 21,
                '22': 22, '23': 23, '24': 24, '25': 25, '26': 26, '27': 27,
                '28': 28, '29': 29, '30': 30, '31': 31, '32': 32, '33': 33,
                '34': 34, '35': 35, '36': 36, '37': 37, '38': 38, '39': 39,
                '40': 40
            }
        deploy_dic_agent_conf = getattr(config, "DIC_BASE_AGENT_CONF")
        deploy_dic_traffic_env_conf = merge(config.dic_traffic_env_conf, dic_traffic_env_conf_extra)
        deploy_dic_path = merge(config.DIC_PATH, dic_path_extra)

        if not os.path.exists("./GPT_logs"):
            os.makedirs("./GPT_logs")

        if in_args.multi_process:


            ppl = Process(target=pipeline_wrapper,
                          args=(dic_agent_conf_extra,
                                deploy_dic_traffic_env_conf,
                                deploy_dic_path))
            process_list.append(ppl)


        else:
            pipeline_wrapper(dic_agent_conf=dic_agent_conf_extra,
                             dic_traffic_env_conf=deploy_dic_traffic_env_conf,
                             dic_path=deploy_dic_path)

    if in_args.multi_process:
        for i in range(0, len(process_list), in_args.workers):
            i_max = min(len(process_list), i + in_args.workers)
            for j in range(i, i_max):
                print(j)
                print("start_traffic")
                process_list[j].start()
                print("after_traffic")
            for k in range(i, i_max):
                print("traffic to join", k)
                process_list[k].join()
                print("traffic finish join", k)

    return in_args.memo


if __name__ == "__main__":
    args = parse_args()

    main(args)

