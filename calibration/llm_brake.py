import os
import sys
import numpy as np
import re

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../transformers/src"))
print(project_root)
sys.path.insert(0, project_root) 

from transformers import AutoTokenizer, BitsAndBytesConfig, AutoConfig
import torch
import torch.nn.functional as F
from transformers.models.phi3.modeling_phi3 import CustomPhi3ForCausalLM
from transformers.models.llama.modeling_llama import CustomLlamaForCausalLM
from transformers.models.qwen2.modeling_qwen2 import Qwen2ForCausalLM
from transformers.models.deepseek_v3.modeling_deepseek_v3 import DeepseekV3ForCausalLM
from transformers.models.falcon.modeling_falcon import FalconForCausalLM
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import test_slim_ffn
import numpy as np


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
 

class LLM_Brake:
    NONE_GROUP_IDX, POSITIVE_GROUP_IDX, NEGATIVE_GROUP_IDX = -1, 0, 1
    # POSITIVE_GROUP = ['true', 'yes']
    # NEGATIVE_GROUP = ['false', 'no']
    POSITIVE_GROUP = ['true']
    NEGATIVE_GROUP = ['false']
    CALIBRATION_MODE = 0
    BRAKE_MODE = 1
    RECORD_MODE = 2
    def __init__(self):
        self.mode = LLM_Brake.CALIBRATION_MODE
        self.model = None
        self.model_name = None
        self.tokenizer = None
        self.threshold = 0.5
        self.k = 10
         
        self.record_exit_layer = []
        self.record_accuracy = []
        self.record = []
        self.current_idx = -1
        self.exit_token = None
        self.last_token = None
        self.next_check_layer = 0
    

    def load(self, model_name, quantization=None):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.true_tok, self.false_tok = self.tokenizer.encode("True", add_special_tokens=False)[0], self.tokenizer.encode("False", add_special_tokens=False)[0]
        if quantization == 4:
            bnb_config = BitsAndBytesConfig(
                load_in_8bit=True
            )
        elif quantization == 8:
            bnb_config = BitsAndBytesConfig(
                load_in_8bit=True
            )
        else:
            bnb_config = None

        config = AutoConfig.from_pretrained(model_name)  
        # print(f'config.num_hidden_layers {config.num_hidden_layers}')
        config.output_hidden_states = True
        self.k_stats = {
            "count": [0] * config.num_hidden_layers,
            "sum": [0] * config.num_hidden_layers,
        }

        self.threshold_stats = {
            "count": [0] * config.num_hidden_layers,
            "sum" : [0] * config.num_hidden_layers,

        }
        
        # todo FIx this

        if 'phi' in model_name.lower():
            self.causalLM = CustomPhi3ForCausalLM.from_pretrained(
                model_name,
                config=config,
                device_map="auto",
                quantization_config=bnb_config,
            )
        elif 'llama' in model_name.lower() or 'tulu' in model_name.lower() or 'yi' in model_name.lower():
            device_map = {

                    "model.embed_tokens": "cuda:0",

                    # GPU0: layers 0-26
                    **{f"model.layers.{i}": "cuda:0" for i in range(0, 27)},

                    # GPU1: layers 27-53
                    **{f"model.layers.{i}": "cuda:1" for i in range(27, 54)},

                    # GPU2: layers 54-79
                    **{f"model.layers.{i}": "cuda:2" for i in range(54, 80)},

                    "model.norm": "cuda:2",
                    "lm_head": "cuda:2",
            }
            self.causalLM = CustomLlamaForCausalLM.from_pretrained(
                model_name,
                config=config,
                device_map=device_map,
                # offload_folder="offload",
                quantization_config=bnb_config,  # needed if using CPU
                torch_dtype="bfloat16"     
            )
            import gc
            self.causalLM = test_slim_ffn.slim_model_tail(self.causalLM, keep_ratio=0.4, start_frac=0.7)
            # self.causalLM.to("cuda")

            gc.collect()
            torch.cuda.empty_cache()
        elif 'qwen' in model_name.lower():
            self.causalLM = Qwen2ForCausalLM.from_pretrained(
                model_name,
                config=config,
                device_map="auto",
                quantization_config=bnb_config,
            )
        elif 'deepseek' in model_name.lower():
            self.causalLM = DeepseekV3ForCausalLM.from_pretrained(
                model_name,
                config=config,
                device_map="auto",
                quantization_config=bnb_config,
            )
        elif 'falcon' in model_name.lower():
            self.causalLM = FalconForCausalLM.from_pretrained(
                model_name,
                config=config,
                device_map="auto",
                quantization_config=bnb_config,
            )
        else:
            raise Exception(f"No Model interface found for {model_name}")
        
        if 'falcon' in model_name.lower():
            self.model_name = model_name
            self.last_layer = config.num_hidden_layers
            self.causalLM.transformer.llm_brake = self
            self.model = self.causalLM.transformer
            self.model.norm = self.model.ln_f
        else:
            self.model_name = model_name
            self.last_layer = config.num_hidden_layers
            self.causalLM.model.llm_brake = self
            self.model = self.causalLM.model
            

        # print(f'self.causalLM.hf_device_map {self.causalLM.hf_device_map}')
        # # Suppose `model` is already loaded
        # for name, param in self.causalLM.model.named_parameters():
        #     print(name, param.device)

        # for name, buffer in self.causalLM.model.named_buffers():
        #     print(name, buffer.device)
        self.group_classifier, self.group_tokenizer = self.init_group_classifer()


    def set_layer_listener(self, listener):
        self.model.set_layer_listener(listener)
         
    def listen_layer(self, hidden_states, idx):
        if self.mode == LLM_Brake.RECORD_MODE:
            self.record_layer(hidden_states[:, -1, :], idx)
            return False
        elif self.mode == self.BRAKE_MODE and self.check_early_exit(hidden_states[:, -1, :], idx):
            self.next_check_layer = 0
            return True
        return False
 

    def normalize(self, token):
        return re.sub(r"[^a-zA-Z0-9]", "", token.strip().lower())

    def init_group_classifer(self):
        model_name = "roberta-large-mnli"
        group_tokenizer = AutoTokenizer.from_pretrained(model_name)
        group_classifier = AutoModelForSequenceClassification.from_pretrained(model_name)
        group_classifier.eval()
        return group_classifier, group_tokenizer

    def compare_words(self, w1: str, w2: str):
        id2label =  self.group_classifier.config.id2label  # {0: 'CONTRADICTION', 1: 'NEUTRAL', 2: 'ENTAILMENT'}
        premise = f"{w1}"
        hypothesis = f"{w2}"
        enc = self.group_tokenizer(premise, hypothesis, return_tensors="pt", truncation=True)
        with torch.no_grad():
            logits = self.group_classifier(**enc).logits
        probs = F.softmax(logits, dim=-1)[0]
        return probs[2]
    
    def get_group_id(self, token):
        group_idx = LLM_Brake.NONE_GROUP_IDX
        if (self.normalize(token) in LLM_Brake.POSITIVE_GROUP):
            group_idx = LLM_Brake.POSITIVE_GROUP_IDX
        elif (self.normalize(token) in LLM_Brake.NEGATIVE_GROUP): 
            group_idx = LLM_Brake.NEGATIVE_GROUP_IDX
        return group_idx

    def get_k_size(self, top_tokens):
        k = 1
        group_idx = self.get_group_id(top_tokens[0])
        if group_idx == LLM_Brake.NONE_GROUP_IDX:
            return -1, -1

        for idx in range(1, len(top_tokens)):
            if self.normalize(top_tokens[idx]) in LLM_Brake.POSITIVE_GROUP and group_idx == LLM_Brake.POSITIVE_GROUP_IDX:
                k += 1
            elif self.normalize(top_tokens[idx]) in LLM_Brake.NEGATIVE_GROUP and group_idx == LLM_Brake.NEGATIVE_GROUP_IDX:
                k += 1
            else:
                return k, group_idx
        return k, group_idx
             

    def get_probs(self, logits, top_tokens=None):
        pos_logit_list = []
        neg_logit_list = []
        for pos_tok in LLM_Brake.POSITIVE_GROUP:
            pos_enc = self.tokenizer.encode(pos_tok, add_special_tokens=False)[0]
            pos_logit_list.append(logits[0, pos_enc].item())

        for neg_tok in LLM_Brake.NEGATIVE_GROUP:
            neg_enc = self.tokenizer.encode(neg_tok, add_special_tokens=False)[0]
            neg_logit_list.append(logits[0, neg_enc].item())
        probs = F.softmax(torch.tensor(pos_logit_list+neg_logit_list), dim=-1)
        mid_idx = len(LLM_Brake.POSITIVE_GROUP)
        return [sum(probs[:mid_idx]), sum(probs[mid_idx:])]


    def get_top_tokens(self, logits, k):
        topK = torch.topk(logits[0], k=k)
        top_tokens = [self.normalize(self.tokenizer.decode([tok])) for tok in topK.indices.tolist()]
        if len(list(filter(None, top_tokens))) == 0:
            print('top token', top_tokens)
        return list(filter(None, top_tokens))

    def get_hidden_states_list(self, statement_list):
        hidden_states_list = []
        for prompt in statement_list: 
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.causalLM.device)
            with torch.no_grad():
                outputs = self.causalLM(**inputs)
                hidden_states = outputs.hidden_states
                hidden_states_list.append(hidden_states)
        return hidden_states_list
    

    def calibrate(self, golden_set):
        if self.mode == LLM_Brake.CALIBRATION_MODE:
            hidden_states_list = self.get_hidden_states_list(golden_set)
            self.calibrate_K(hidden_states_list)
            self.calibrate_threshold(hidden_states_list)

    @torch.no_grad()
    def calibrate_K(self, hidden_states_list): 
        start_layer = int(self.last_layer * 0)
        end_layer = int(self.last_layer * 1)

        # calibrate k
        for hidden_states in hidden_states_list:
            for layer_idx, h in enumerate(hidden_states[start_layer:end_layer]):

                logits = self.causalLM.lm_head(self.model.norm(h)[:, -1, :])
                top_tokens = self.get_top_tokens(logits, 10)
                print(top_tokens)
                k, _ = self.get_k_size(top_tokens)
                # print(top_tokens)
                if k > 0:
                    self.k_stats["count"][start_layer+layer_idx] += 1
                    self.k_stats["sum"][start_layer+layer_idx] += k
        avg_k = np.array(self.k_stats["sum"]) / np.array(self.k_stats["count"])
        avg_k = int(np.nanmean(avg_k.tolist()))
                    
        if avg_k > 5:
            self.k = 5
        else:
            self.k = avg_k
        print(f'\n========== CALIBRATION ============')
        print(f'Model       : {self.model_name}')
        print(f'K           : {self.k}')
        print(f'===================================\n')
    

    @torch.no_grad()
    def calibrate_threshold(self, hidden_states_list): 
        start_layer = int(self.last_layer* 0)
        end_layer = int(self.last_layer* 1)
        avg_threshold = -1
        # calibrate Threshold
        for hidden_states in hidden_states_list:
            for layer_idx, h in enumerate(hidden_states[start_layer:end_layer]):

                logits = self.causalLM.lm_head(self.model.norm(h)[:, -1, :])
                top_tokens = self.get_top_tokens(logits, 10)
                k, group_id = self.get_k_size(top_tokens)
                probs = self.get_probs(logits, top_tokens)
                prob = probs[0] if group_id == 0 else probs[1]

                if k > 0 :
                    # print(f'layer_idx, p, k {layer_idx, prob, k}')
                    self.threshold_stats["count"][start_layer+layer_idx] += 1
                    self.threshold_stats["sum"][start_layer+layer_idx] += prob
        avg_threshold_per_layer = np.array(self.threshold_stats["sum"]) / np.array(self.threshold_stats["count"])
        avg_threshold_per_layer = remove_outliers_keep_nans(avg_threshold_per_layer)
        avg_threshold = float(np.nanmean(avg_threshold_per_layer.tolist()))

        if avg_threshold > 0.99:
            self.threshold = 0.99
        else:
            self.threshold = avg_threshold
        
        print(f'\n========== CALIBRATION ============')
        print(f'Model       : {self.model_name}')
        print(f'Threshold   : {self.threshold}')
        print(f'===================================\n')
    
    
    def layer_range(self, layer_id, start_layer, interval):
        if (layer_id > start_layer and layer_id % interval == 0):
            return True
        return False

    @torch.no_grad()
    def record_layer(self, hidden_state, layer_id=None):
        # Check available layer range
        interval = int(0.1 * self.last_layer) if int(0.1 * self.last_layer) > 1 else 2
         
        if self.layer_range(layer_id, start_layer=0, interval=1):
            # Check conditions
            logits = self.causalLM.lm_head(hidden_state)
            
            top_tokens = self.get_top_tokens(logits, 10)
            probs = self.get_probs(logits, top_tokens)
            prob = probs[0] if probs[0] > probs[1] else probs[1]

            normalized_top_tokens = list(map(lambda tok: self.normalize(tok), top_tokens[:self.k]))
            group_ids = list(map(lambda tok: self.get_group_id(tok), top_tokens[:self.k]))
            # Last layer
            if layer_id == self.last_layer-1:
                self.last_token = normalized_top_tokens[0]
                if self.exit_token is None:
                    precision = -1
                elif self.get_group_id(self.last_token) == self.get_group_id(self.exit_token):
                    precision = 1
                elif self.exit_token is not None:
                    precision = 0
                self.exit_token = None

                record = {
                    'idx': self.current_idx,
                    'layer_id': layer_id,
                    'last_layer_id': layer_id,
                    'top_tokens': top_tokens,
                    'probs' : probs,
                    'precision': precision
                }

            # Set exit_token if satisfying conditions
            elif len(set(group_ids)) == 1 \
                and (prob > self.threshold) \
                and ((group_ids[0] == LLM_Brake.POSITIVE_GROUP_IDX and probs[0] > probs[1]) 
                     or (group_ids[0] == LLM_Brake.NEGATIVE_GROUP_IDX and probs[0] < probs[1])) \
                and self.exit_token is None \
                and self.layer_range(layer_id, int(0.5 * self.last_layer), interval):

                self.exit_token = normalized_top_tokens[0]
                
                record = {
                    'idx': self.current_idx,
                    'layer_id': layer_id,
                    'exit_layer_id': layer_id,
                    'top_tokens': top_tokens,
                    'probs' : probs,
                }
            else:
                record = {
                    'idx': self.current_idx,
                    'layer_id': layer_id,
                    'top_tokens': top_tokens,
                    'probs' : probs,
                }
            self.record.append(record)

    @torch.no_grad()
    def check_early_exit(self, hidden_state, layer_id=None):
        # Check available layer range
        interval = int(0.1 * self.last_layer) if int(0.1 * self.last_layer) > 1 else 2
        if not self.layer_range(layer_id, int(0.5 * self.last_layer), interval):
            return False
        # Check conditions
        logits = self.causalLM.lm_head(hidden_state)

        probs = self.get_probs(logits)
        prob = probs[0] if probs[0] > probs[1] else probs[1]
        top_tokens = self.get_top_tokens(logits, self.k)
        group_ids = list(map(lambda tok: self.get_group_id(tok), top_tokens))

        # Condition 1. prob > threshold
        if prob < self.threshold:
            return False
        # Condition 2. top k token
        if len(set(group_ids)) == 1 and (group_ids[0] == LLM_Brake.POSITIVE_GROUP_IDX and probs[0] > probs[1]) or \
            (group_ids[0] == LLM_Brake.NEGATIVE_GROUP_IDX and probs[0] < probs[1]):
            return True
        
    @torch.no_grad()
    def inference(self, prompt, current_idx):
        self.current_idx = current_idx
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.causalLM.device)
        with torch.no_grad():
            outputs = self.causalLM(**inputs)
        hidden_state = outputs.hidden_states
        logits = self.causalLM.lm_head(hidden_state[-1][:, -1, :])
        topk_tokens = self.get_top_tokens(logits, 3)
        return self.normalize(topk_tokens[0])
    
