
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
from accelerate.hooks import remove_hook_from_module
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache


torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_math_sdp(True)

model_name = "/datasets/ai/llama3/hub/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659"
model_name = "/datasets/ai/llama3/hub/models--meta-llama--Meta-Llama-3-70B/snapshots/c82494877ce7f6d7d317c56ec081328e382c72fe"
model_name = "/datasets/ai/llama3/hub/models--meta-llama--Llama-3.2-1B/snapshots/4e20de362430cd3b72f300e6b0f18e50e7166e08"
tokenizer = AutoTokenizer.from_pretrained(model_name)

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map='auto',
    attn_implementation="eager"
)
model.eval()
for module in model.modules():
    remove_hook_from_module(module)
import torch

# Document A
A = "Hello world!"
# Query q
q = "Test hi"

A_ids = tokenizer(A, return_tensors="pt").to(model.device)
q_ids = tokenizer(q, add_special_tokens=False, return_tensors="pt").to(model.device)
# ---- 1) Build KV cache on A ----
with torch.no_grad():
    out_A = model(**A_ids, use_cache=True, return_dict=True)
a_past = out_A.past_key_values

# A_len = A_ids["input_ids"].shape[1]

# full_mask = torch.cat(
#     [A_ids["attention_mask"], q_ids["attention_mask"]],
#     dim=1
# )
# q_step = {
#     "input_ids": q_ids["input_ids"],
#     "attention_mask": full_mask
# }

# with torch.no_grad():
#     out_q = model(
#         **q_step,
#         past_key_values=past,
#         use_cache=True,
#         return_dict=True
#     )


def get_first_part_kv_cache(first, second):
    first_ids = tokenizer(first, return_tensors="pt").to(model.device)
    first_len = first_ids["input_ids"].shape[1]
    prompt = first + second
    full = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        out_full = model(
            **full,
            use_cache=True,
            return_dict=True
        )
    new_past = []

    for layer in range(len(out_full.past_key_values)):
        k, v = out_full.past_key_values[layer]
        # slice sequence dim (dim=2)
        k_new = k[:, :, :first_len, :]   # first 4 tokens
        v_new = v[:, :, :first_len, :]
        new_past.append((k_new, v_new))

    return DynamicCache.from_legacy_cache(new_past)

a_cache = get_first_part_kv_cache(A, q)

for i in range(16):
    for j in range(2):
        print(i,j, (a_past[i][j] == a_cache[i][j]).all())
print()
