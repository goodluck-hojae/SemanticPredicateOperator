
import utils
import os
import json
import argparse
from llm_brake import LLM_Brake
from Data import biodex, fever, arxiv, boolq
import time
import utils


parser = argparse.ArgumentParser()
parser.add_argument(
    "--model_name",
    type=str,
    # default='/datasets/ai/llama3/hub/models--meta-llama--Llama-3.2-3B-Instruct/snapshots/0cb88a4f764b7a12671c53f0838cd831a0843b95'
    # default='/datasets/ai/llama3/hub/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659',
    # default='/datasets/ai/qwen/hub/models--Qwen--Qwen2.5-32B-Instruct/snapshots/5ede1c97bbab6ce5cda5812749b4c0bdf79b18dd'
    # default='/datasets/ai/qwen/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28'
    # default='/datasets/ai/phi/hub/models--microsoft--Phi-3.5-mini-instruct/snapshots/3145e03a9fd4cdd7cd953c34d9bbf7ad606122ca', 
    # default='/datasets/ai/falcon/hub/models--tiiuae--falcon-40b/snapshots/05ab2ee8d6b593bdbab17d728de5c028a7a94d83'
    # default='/datasets/ai/yi/hub/models--01-ai--Yi-34B/snapshots/e1fa7c83283f2e1d5a12cb4adea37c3f829e28f8'
    default= '/datasets/ai/llama3/hub/models--meta-llama--Llama-3.3-70B-Instruct/snapshots/6f6073b423013f6a7d4d9f39144961bfbfbc386b'
)
parser.add_argument(
    "--quantization",
    type=int,
    default=None,
    help="Quantization bit width (e.g., 4, 8)"
)

parser.add_argument(
    "--start_idx",
    type=int,
    default=0,
)
args = parser.parse_args()


def run(start_idx):
    # gen = fever.yield_llm_statement("/home/hojaeson_umass_edu/project/transformers/src/transformers/research/early_exit_sem_join/Data/feverous_dev_challenges.jsonl", 200)
    # gen = arxiv.yield_arxiv_contradiction_statement()
    
    gen = boolq.iter_boolq_validation(max_iter=1000)

    start = time.time()
    total = 0
    idx = 0
    latency_list = []
    accuracy_list = []
    while True:
        case_start = time.time()
        try:
            statement, label = next(gen)
            label = _llm_brake.normalize(str(label))
            if idx >= start_idx:
                normal_top_token = _llm_brake.inference(statement, idx)
                # print('normal_top_token, label', normal_top_token, label)
                accuracy_list.append(normal_top_token == label)
            if idx % 100 == 0:
                print(f'{idx} In progress..')
        except StopIteration:
            break
        
        latency = time.time() - case_start
        total += latency
        latency_list.append(latency)
        idx += 1

    end = time.time()
    print(f'accuracy {accuracy_list.count(True)/len(accuracy_list)}')
    print(f'total time {end - start}')
    return latency_list, accuracy_list


def get_golden_set(path="/home/hojaeson_umass_edu/project/transformer_research/early_exit_sem_join/Data/golden_set.json"):
    golden_set = []
    with open(path) as f:
        data = json.load(f)
        for idx, row in enumerate(data):
            evidence, claim = row["evidence"], row["claim"]
            statement = fever.construct_statement(evidence, claim)
            golden_set.append(statement)
    return golden_set


if __name__ == '__main__':
    _llm_brake = LLM_Brake()
    _llm_brake.load(args.model_name, args.quantization)

    golden_set = get_golden_set()
    print('Calibrating..')
    _llm_brake.mode = LLM_Brake.CALIBRATION_MODE
    _llm_brake.calibrate(golden_set)

    print('======================Start Braking mode')
    _llm_brake.mode = LLM_Brake.BRAKE_MODE
    _llm_brake.set_layer_listener(_llm_brake.listen_layer)
    braking_latency, braking_accuracy_list = run(args.start_idx)

    print('======================Start Normal mode')
    _llm_brake.set_layer_listener(None)
    normal_latency, normal_accuracy_list = run(args.start_idx)
    
    print('======================Start Record mode')
    _llm_brake.mode = LLM_Brake.RECORD_MODE
    _llm_brake.set_layer_listener(_llm_brake.listen_layer)
    run(args.start_idx)
    print(len(_llm_brake.record))
    # for idx in range(0, len(normal_latency)):
    #     _llm_brake.record[idx*(_llm_brake.last_layer-1)]['exit_time'] = braking_latency[idx]
    #     _llm_brake.record[idx*(_llm_brake.last_layer-1)]['normal_time'] = normal_latency[idx]
    utils.save_record(args.model_name, _llm_brake.record)
    print('braking_accuracy_list', braking_accuracy_list)
    print('normal_accuracy_list', normal_accuracy_list)
    print('Terminating..')