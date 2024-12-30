from .pipeline import Pipeline
import copy
import os
import json


def merge(dic_tmp, dic_to_change):
    dic_result = copy.deepcopy(dic_tmp)
    dic_result.update(dic_to_change)
    return dic_result


def pipeline_wrapper(dic_agent_conf, dic_traffic_env_conf, dic_path):

    ppl = Pipeline(dic_agent_conf=dic_agent_conf,
                   dic_traffic_env_conf=dic_traffic_env_conf,
                   dic_path=dic_path
                   )
    if dic_traffic_env_conf['MODEL_NAME'] == 'gpt' or dic_traffic_env_conf['MODEL_NAME'] == 'Llama' or dic_traffic_env_conf['MODEL_NAME'] == 'llama3170B':
        ppl.run_gpt(multi_process=False)

    elif dic_traffic_env_conf['MODEL_NAME'] == 'herald':
        ppl.run_herald(multi_process=False)
    else:
        ppl.run(multi_process=False)

    print("pipeline_wrapper end")
    return


