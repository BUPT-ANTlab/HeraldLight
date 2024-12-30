from .config import DIC_AGENTS
from .cityflow_env import CityFlowEnv
import time
import os
import copy
from collections import deque
import numpy as np
import wandb
from models.Agent_plug_in import Agent_plug_in

class Generator:
    def __init__(self, cnt_round, cnt_gen, dic_path, dic_agent_conf, dic_traffic_env_conf):

        self.cnt_round = cnt_round
        self.cnt_gen = cnt_gen
        self.dic_path = dic_path
        self.dic_agent_conf = copy.deepcopy(dic_agent_conf)
        self.dic_traffic_env_conf = dic_traffic_env_conf
        self.agents = [None]*dic_traffic_env_conf['NUM_AGENTS']
        self.path_to_log = os.path.join(self.dic_path["PATH_TO_WORK_DIRECTORY"], "train_round",
                                        "round_"+str(self.cnt_round), "generator_"+str(self.cnt_gen))
        self.Llama_expert = None
        self.wandb_enable = True


        if not os.path.exists(self.path_to_log):
            os.makedirs(self.path_to_log)
            start_time = time.time()
            for i in range(dic_traffic_env_conf['NUM_AGENTS']):
                agent_name = self.dic_traffic_env_conf["MODEL_NAME"]
                if "gpt" in agent_name:
                    agent = DIC_AGENTS[agent_name](
                        dic_agent_conf=self.dic_agent_conf,
                        dic_traffic_env_conf=self.dic_traffic_env_conf,
                        dic_path=self.dic_path,
                        cnt_round=self.cnt_round,
                        intersection_id=str(i),
                        log_dir=self.dic_agent_conf["LOG_DIR"],  # extra ./GPT_logs
                        dataset=f"{self.dic_traffic_env_conf['TRAFFIC_FILE']}"
                    )
                    self.agents[i] = agent
                elif "llama3170B" in agent_name:
                    agent = DIC_AGENTS['gpt'](
                        dic_agent_conf=self.dic_agent_conf,
                        dic_traffic_env_conf=self.dic_traffic_env_conf,
                        dic_path=self.dic_path,
                        cnt_round=self.cnt_round,
                        intersection_id=str(i),
                        log_dir=self.dic_agent_conf["LOG_DIR"],  # extra ./GPT_logs
                        dataset=f"{self.dic_traffic_env_conf['TRAFFIC_FILE']}"
                    )
                    self.agents[i] = agent
                elif"Llama" in agent_name:
                    agent = DIC_AGENTS[agent_name](
                        dic_agent_conf=self.dic_agent_conf,
                        dic_traffic_env_conf=self.dic_traffic_env_conf,
                        dic_path=self.dic_path,
                        cnt_round=self.cnt_round,
                        intersection_id=str(i),
                        log_dir=self.dic_agent_conf["LOG_DIR"],  # extra ./GPT_logs
                        dataset=f"{self.dic_traffic_env_conf['TRAFFIC_FILE']}"
                    )
                    self.agents[i] = agent
                    # self.Llama_expert = DIC_AGENTS["Llama_expert"](
                    #     agent.Llama_model, agent.tokenizer, agent.sampling_params,self.dic_traffic_env_conf,self.dic_agent_conf["LOG_DIR"],
                    # )
                else:
                    agent = DIC_AGENTS[agent_name](
                        dic_agent_conf=self.dic_agent_conf,
                        dic_traffic_env_conf=self.dic_traffic_env_conf,
                        dic_path=self.dic_path,
                        cnt_round=self.cnt_round,
                        intersection_id=str(i),
                    )
                    self.agents[i] = agent
                    self.wandb_enable = False



            print("Create intersection agent time: ", time.time()-start_time)
        self.env = CityFlowEnv(
            path_to_log=self.path_to_log,
            path_to_work_directory=self.dic_path["PATH_TO_WORK_DIRECTORY"],
            dic_traffic_env_conf=self.dic_traffic_env_conf
        )


    def generate(self):

        if self.wandb_enable :
            wandb.init(project=f"{self.dic_traffic_env_conf['MODEL_NAME']}", name="run_{}_{}".format(self.dic_traffic_env_conf['WANDB_NAME'], self.dic_traffic_env_conf['MODEL_NAME']))
        reset_env_start_time = time.time()
        state, step_time, list_need = self.env.reset()
        reset_env_time = time.time() - reset_env_start_time
        running_start_time = time.time()

        vehicle_pass_num = {}
        vehicle_pass_num = {inter: 0 for inter in range(self.env.num_intersection)}

        vehicle_pass_num_duration_left = {}
        vehicle_pass_num_duration_left = {inter: [] for inter in range(self.env.num_intersection)}

        history_intersection = {inter: [] for inter in range(self.env.num_intersection)}
        self.env.list_gpt_history = [deque(maxlen=3) for _ in range(self.env.num_intersection)]
        one_time_toggle = False

        queue_length_episode = []
        waiting_time_episode = []
        while step_time < self.dic_traffic_env_conf["RUN_COUNTS"]:
            if self.wandb_enable:
                wandb.log({'step_time' : step_time})
            step_start_time = time.time()
            phase_action_plug_in = []
            phase_action_plug_in = [0] * len(phase_action_plug_in)

            #agent只有一个
            for i in range(self.dic_traffic_env_conf["NUM_AGENTS"]):

                #agent给出动作，这里的Agent是[<models.dynamiclight.DynamicLightAgent object at 0x71b881f5d5a0>]
                if self.dic_traffic_env_conf['MODEL_NAME'] == 'gpt' or self.dic_traffic_env_conf['MODEL_NAME'] == 'Llama' or self.dic_traffic_env_conf['MODEL_NAME'] == 'llama3170B':
                    if self.dic_traffic_env_conf['MODEL_NAME'] == 'Llama' :
                        phase_action, duration_action, _ = self.agents[i].choose_action(state,
                                                                                                                 list_need,
                                                                                                                 self.env.list_intersection,
                                                                                                                 self.env.list_gpt_history,
                                                                                                                 phase_action_plug_in,
                                                                                                                 step_time)

                    else:
                        phase_action, duration_action = self.agents[i].choose_action(state, list_need, self.env.list_intersection, self.env.list_gpt_history)

                elif self.dic_traffic_env_conf['MODEL_NAME'] == 'herald':
                    phase_action, duration_action = self.agents[i].choose_action(state, list_need, self.env.list_intersection,self.env.actions_static, self.env.vehicle_pass_num_duration_left_for_herald,step_time)

                else:
                    phase_action, duration_action = self.agents[i].choose_action(state, list_need)


            if self.dic_traffic_env_conf['MODEL_NAME'] == 'gpt' or self.dic_traffic_env_conf['MODEL_NAME'] == 'Llama' or self.dic_traffic_env_conf['MODEL_NAME'] == 'llama3170B':


                next_state, step_time, list_need = self.env.step_gpt_5yellow( phase_action, duration_action, vehicle_pass_num, history_intersection, vehicle_pass_num_duration_left)

            elif self.dic_traffic_env_conf['MODEL_NAME'] == 'herald':
                next_state, step_time, list_need = self.env.step_gpt_5yellow(phase_action, duration_action,
                                                                             vehicle_pass_num, history_intersection,
                                                                             vehicle_pass_num_duration_left)

            else:
                next_state, step_time, list_need = self.env.step(phase_action, duration_action)

            print("after one step time: {0}, one step running_time: {1}".format(self.env.get_current_time(),
                                                        time.time()-step_start_time))
            print('step_time',step_time,"\n")



            #queue_length
            queue_length_inter = []
            for inter in self.env.list_intersection:
                queue_length_inter.append(sum(inter.dic_feature['lane_queue_vehicle_in']))
            queue_length_episode.append(sum(queue_length_inter))

            # waiting time
            waiting_times = []
            for veh in self.env.waiting_vehicle_list:
                waiting_times.append(self.env.waiting_vehicle_list[veh]['time'])
            waiting_time_episode.append(np.mean(waiting_times) if len(waiting_times) > 0 else 0.0)
            state = next_state

        running_time = time.time() - running_start_time
        log_start_time = time.time()
        print("start logging.......................")
        self.env.bulk_log_multi_process()
        log_time = time.time() - log_start_time
        self.env.end_cityflow()
        print("reset_env_time: ", reset_env_time)
        print("running_time_total: ", running_time)
        print("log_time: ", log_time)
        ################################################33
        vehicle_travel_times = {}
        for inter in self.env.list_intersection:
            arrive_left_times = inter.dic_vehicle_arrive_leave_time
            for veh in arrive_left_times:
                enter_time = arrive_left_times[veh]["enter_time"]
                leave_time = arrive_left_times[veh]["leave_time"]
                if not np.isnan(enter_time) and not np.isnan(leave_time):
                    if veh not in vehicle_travel_times:
                        vehicle_travel_times[veh] = [leave_time - enter_time]
                    else:
                        vehicle_travel_times[veh].append(leave_time - enter_time)

        total_travel_time = np.mean([sum(vehicle_travel_times[veh]) for veh in vehicle_travel_times])
        #print('total_travel_time\n\n\n',total_travel_time)

        results = {
            "training_avg_travel_time": total_travel_time,
            "training_avg_queue_len": np.mean(queue_length_episode) if len(queue_length_episode) > 0 else 0,
            "training_avg_waiting_time": np.mean(waiting_time_episode) if len(queue_length_episode) > 0 else 0,
        }
        print(results)

        log_file_path = os.path.join(self.path_to_log, 'log.txt')

        with open(log_file_path, 'a') as f:
            f.write(f"{results}\n")

        if self.wandb_enable:

            wandb.log(results)
            wandb.finish()
