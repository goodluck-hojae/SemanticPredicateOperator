import os, sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../transformers/src"))
print(project_root)
sys.path.insert(0, project_root) 

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
src_path = os.path.join(project_root, "backend")
if src_path not in sys.path:
    sys.path.append(src_path)


import torch
import time
from hidden import Hidden
from pool import LayerwiseHiddenPool
from model import LayerManager
from controller import PipelineController
from accelerate.hooks import remove_hook_from_module
from transformers import AutoModelForCausalLM, AutoTokenizer 
 
import numpy as np
import re
import torch.nn.functional as F


def normalize(token):
    return re.sub(r"[^a-zA-Z0-9]", "", token.strip().lower())

def remove_outliers_keep_nans(arr):
    arr = np.array(arr, dtype=float)
    mask_nan = np.isnan(arr)
    vals = arr[~mask_nan]
    cutoff = 1.5
    q1, q3 = np.percentile(vals, [25, 75])
    iqr = q3 - q1
    lower_bound = q1 - cutoff * iqr
    upper_bound = q3 + cutoff * iqr
    outlier_mask = (arr < lower_bound) | (arr > upper_bound)
    outlier_mask[mask_nan] = False
    arr[outlier_mask] = np.nan
    return arr
 

class Calibration:
    NONE_GROUP_IDX, POSITIVE_GROUP_IDX, NEGATIVE_GROUP_IDX = -1, 0, 1
    # POSITIVE_GROUP = ['true', 'yes']
    # NEGATIVE_GROUP = ['false', 'no']
    POSITIVE_GROUP = ['true']
    NEGATIVE_GROUP = ['false']
    CALIBRATION_MODE = 0
    BRAKE_MODE = 1
    RECORD_MODE = 2
    def __init__(self, layer_manager):
        self.layer_manager = layer_manager
         
        self.k_stats = {
            "count": [0] * layer_manager.num_layers(),
            "sum": [0] * layer_manager.num_layers(),
        }

        self.threshold_stats = {
            "count": [0] * layer_manager.num_layers(),
            "sum" : [0] * layer_manager.num_layers(),
        }

        self.k = -1
        self.threshold = -1

    def check_exit(self, hidden_state):
        top_tokens = self.top_tokens(hidden_state)
        
        layer_k, group_id = self.get_k_size(top_tokens)
        prob = self.get_probs(hidden_state)
        layer_prob = prob[0] if group_id == 0 else prob[1]
        if layer_k >= self.k and layer_prob >= self.threshold:
            return True
        return False
    
        # test with exiting at random layers
        import random
        x = random.choice([40, 42, 44, 48, 50, 55, 60, 75])
        if hidden.layer_id == x:
            hidden.prediction_token = top_tokens
            hidden.exit_layer = hidden.layer_id
            return True


    def exit_params(self):
        return {"k":self.k, "threshold":self.threshold}

    def get_group_id(token):
        group_idx = Calibration.NONE_GROUP_IDX
        if (normalize(token) in ['true']):
            group_idx = Calibration.POSITIVE_GROUP_IDX
        elif (normalize(token) in ['false']): 
            group_idx = Calibration.NEGATIVE_GROUP_IDX
        return group_idx


    def get_k_size(self, top_tokens):
        k = 1
        group_idx = Calibration.get_group_id(top_tokens[0])
        
        for idx in range(1, len(top_tokens)):
            if normalize(top_tokens[idx]) in ['true']:
                k += 1
            elif normalize(top_tokens[idx]) in ['false']:
                k += 1
            else:
                return k, group_idx
        return k, group_idx
            

    @torch.no_grad()
    def get_probs(self, hidden):
        layer_manager = self.layer_manager
        logits =layer_manager.lm_head(layer_manager.norm(hidden)[:, -1, :])
        pos_logit_list = []
        neg_logit_list = []
        for pos_tok in ['true']:
            pos_enc = layer_manager.tokenizer.encode(pos_tok, add_special_tokens=False)[0]
            pos_logit_list.append(logits[0, pos_enc].item())

        for neg_tok in ['false']:
            neg_enc = layer_manager.tokenizer.encode(neg_tok, add_special_tokens=False)[0]
            neg_logit_list.append(logits[0, neg_enc].item())
        probs = F.softmax(torch.tensor(pos_logit_list+neg_logit_list), dim=-1)
        mid_idx = 1
        return [sum(probs[:mid_idx]), sum(probs[mid_idx:])]



 
    @torch.no_grad()
    def top_tokens(self, hidden_state):
        layer_manager = self.layer_manager
        logits = layer_manager.lm_head(layer_manager.norm(hidden_state)[:, -1, :])
        topK = torch.topk(logits[0], k=10)
        top_tokens = [layer_manager.tokenizer.decode([tok]) for tok in topK.indices.tolist()]
        return top_tokens


    @torch.no_grad()
    def calibrate_K(self, hidden_states_list): 
        layer_manager = self.layer_manager
        start_layer = int(layer_manager.num_layers() * 0.7)
        end_layer = int(layer_manager.num_layers() * 1)

        # calibrate k
        for hidden_states in hidden_states_list:
            for layer_idx, h in enumerate(hidden_states[start_layer:end_layer]):

                top_tokens = self.top_tokens(h)
                k, _ = self.get_k_size(top_tokens)
                # print(top_tokens)
                if k > 0:
                    self.k_stats["count"][start_layer+layer_idx] += 1
                    self.k_stats["sum"][start_layer+layer_idx] += k
        avg_k = np.array(self.k_stats["sum"]) / np.array(self.k_stats["count"])
        avg_k = int(np.nanmean(avg_k.tolist()))
        print(np.array(self.k_stats["sum"]))
        print(np.array(self.k_stats["count"]))
        print(np.array(self.k_stats["sum"]) / np.array(self.k_stats["count"]))
        if avg_k > 5:
            self.k = 5
        else:
            self.k = avg_k
        print(f'\n========== CALIBRATION ============')
        print(f'K           : {k}')
        print(f'===================================\n')


    @torch.no_grad()
    def calibrate_threshold(self, hidden_states_list): 
        layer_manager = self.layer_manager
        start_layer = int(layer_manager.num_layers() * 0.7)
        end_layer = int(layer_manager.num_layers() * 1)
        avg_threshold = -1
        # calibrate Threshold
        for hidden_states in hidden_states_list:
            for layer_idx, h in enumerate(hidden_states[start_layer:end_layer]):

                top_tokens = layer_manager.top_tokens(h)
                k, group_id = self.get_k_size(top_tokens)

                probs = self.get_probs(h)
                prob = probs[0] if group_id == 0 else probs[1]

                if k > 0 :
                    self.threshold_stats["count"][start_layer+layer_idx] += 1
                    self.threshold_stats["sum"][start_layer+layer_idx] += prob
        avg_threshold_per_layer = np.array(self.threshold_stats["sum"]) / np.array(self.threshold_stats["count"])
        avg_threshold_per_layer = remove_outliers_keep_nans(avg_threshold_per_layer)
        avg_threshold = float(np.nanmean(avg_threshold_per_layer.tolist()))

        print(self.threshold_stats["sum"]), '\n',  np.array(self.threshold_stats["count"])
        print(avg_threshold_per_layer.tolist())
        if avg_threshold > 0.99:
            self.threshold = 0.99
        else:
            self.threshold = avg_threshold
        
        print(f'\n========== CALIBRATION ============')
        print(f'Threshold   : {self.threshold}')
        print(f'===================================\n')


# test code
if __name__ == '__main__':
    
    import os, sys
    from pool import LayerwiseHiddenPool
    from accelerate.hooks import remove_hook_from_module

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../transformers/src"))
    print(project_root)
    sys.path.insert(0, project_root) 

    from transformers import AutoModelForCausalLM, AutoTokenizer
    model_name = "/datasets/ai/llama3/hub/models--meta-llama--Meta-Llama-3-70B/snapshots/c82494877ce7f6d7d317c56ec081328e382c72fe"
    model_name = '/datasets/ai/llama3/hub/models--meta-llama--Llama-3.2-1B/snapshots/4e20de362430cd3b72f300e6b0f18e50e7166e08'
    model_name='/datasets/ai/llama3/hub/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659'
    tok = AutoTokenizer.from_pretrained(model_name)

    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map={
            "model.embed_tokens": "cuda",
            **{f"model.layers.{i}": "cpu" for i in range(80)},
            "model.norm": "cuda",
            "lm_head": "cuda",
        },
    )
    model.eval()
    for module in model.modules():
        remove_hook_from_module(module)

    pool = LayerwiseHiddenPool()
    layer_manager = LayerManager(model, tok, pool, device="cuda")
    calibrator = Calibration(layer_manager)

    true_false_items = [
    ("A transformer model processes tokens in parallel. True or false?", True),
    ("Cross-entropy loss is commonly used for classification tasks. True or false?", True),
    ("Softmax outputs do not sum to one. True or false?", False),
    ("Dropout is used to reduce overfitting. True or false?", True),
    ("The KV cache speeds up autoregressive decoding. True or false?", True),
    ("Batch normalization is standard in transformer architectures. True or false?", False),
    ("FP16 has higher precision than FP32. True or false?", False),
    ("In self-attention, queries attend only to external input, never to themselves. True or false?", False),
    ("Overfitting means performing well on training data but poorly on unseen data. True or false?", True),
    ("In causal language models, future tokens are masked during training. True or false?", True),
    ]
  
    print("Collecting sample hiddens...")
    hidden_states_list = []
    for i in range(len(true_false_items)):
        prompt = true_false_items[i][0]
        input_ids = tok(prompt, return_tensors="pt").to("cuda")["input_ids"]
        hidden_states, poistion_ids, position_embeddings = layer_manager.process_input_tokens(input_ids, prompt)
        hidden=Hidden(
            id=i,
            states=hidden_states.clone(),
            prompt=prompt,
            pos_ids=poistion_ids,
            pos_emb=position_embeddings,
        )
        full_hidden_states = []
        for layer_id in range(layer_manager.num_layers()):
            next_hidden_list = layer_manager.execute_hiddens([hidden], False)
            top_tokens = calibrator.top_tokens(next_hidden_list[0].states)
            hidden = next_hidden_list[0]

            full_hidden_states.append(hidden.states)
            print(hidden.id, top_tokens, layer_id)
        hidden_states_list.append(full_hidden_states)
        print('len(hidden_states_list)', len(hidden_states_list))
        print('\n')

    print(calibrator.exit_params())
    calibrator.calibrate_K(hidden_states_list)
    calibrator.calibrate_threshold(hidden_states_list)
    print(calibrator.exit_params())