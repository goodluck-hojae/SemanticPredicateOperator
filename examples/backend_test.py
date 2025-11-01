
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

model_name = "/datasets/ai/llama3/hub/models--meta-llama--Llama-3.2-1B/snapshots/4e20de362430cd3b72f300e6b0f18e50e7166e08"
model_name = "/datasets/ai/llama3/hub/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659"
model_name = "/datasets/ai/llama3/hub/models--meta-llama--Meta-Llama-3-70B/snapshots/c82494877ce7f6d7d317c56ec081328e382c72fe"
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

prompt = "How are you?" * 10
input_ids = tok(prompt, return_tensors="pt").to("cuda")["input_ids"]

print("Initializing pool with sample hiddens...")
for i in range(2400):
    hidden_states, poistion_ids, position_embeddings = layer_manager.process_input_tokens(input_ids, prompt)
    pool.store(
        layer_id=0,
        hidden=Hidden(
            id=i,
            states=hidden_states.clone(),
            prompt=prompt,
            pos_ids=poistion_ids,
            pos_emb=position_embeddings,
        ),
    )

print(f"Layer 0 pool initialized with {pool.get_size(0)} hiddens.\n")

controller = PipelineController(
    layer_manager=layer_manager,
    pool=pool,
    min_batch_size=120,
    max_batches_per_layer=2
)

print("=== Starting pipeline loop ===")
start_time = time.time()

step_count = 0
while True:
    step_count += 1
    cont = controller.step()
    if not cont:
        break

print(f"=== Done in {time.time() - start_time:.2f}s after {step_count} steps ===")

for lid, hlist in pool.hidden_states.items():
    print(f"Layer {lid}: {len(hlist)} remaining hiddens")