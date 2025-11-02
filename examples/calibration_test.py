import os, sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../transformers/src"))
print(project_root)
sys.path.insert(0, project_root) 

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
backend_path = os.path.join(project_root, "backend")
calibration_path = os.path.join(project_root, "calibration")
sys.path.extend([backend_path, calibration_path])


import torch
import time
from hidden import Hidden
from pool import LayerwiseHiddenPool
from model import LayerManager
from controller import PipelineController

from accelerate.hooks import remove_hook_from_module
from transformers import AutoModelForCausalLM, AutoTokenizer 
import torch.nn.functional as F
from calibrate import Calibration


def calibrate(calibrator):    
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
        hidden_states_list.append(hidden)



    print("Collecting sample hiddens...")

    N = len(true_false_items)
    L = layer_manager.num_layers()

    # pre-allocate (items × layers)
    hidden_states_list = [[None for _ in range(L)] for _ in range(N)]

    # cache hidden state per item before forward stepping through layers
    initial_hiddens = []
    for i in range(N):
        prompt = true_false_items[i][0]
        input_ids = tok(prompt, return_tensors="pt").to("cuda")["input_ids"]
        hidden_states, position_ids, position_embeddings = layer_manager.process_input_tokens(input_ids, prompt)

        initial_hiddens.append(
            Hidden(
                id=i,
                states=hidden_states.clone(),
                prompt=prompt,
                pos_ids=position_ids,
                pos_emb=position_embeddings,
            )
        )

    # iterate layer-major
    for layer_id in range(L):
    
        if not layer_manager.is_layer_active(layer_id):
            layer_manager.switch_active_layers(start_layer=layer_id)

        new_hidden_list = []
        for i in range(N):
            next_hidden_list = layer_manager.execute_hiddens([initial_hiddens[i]], False)
            hidden = next_hidden_list[0]

            hidden_states_list[i][layer_id] = hidden.states.clone()

            top_tokens = calibrator.top_tokens(hidden.states)
            print(hidden.id, top_tokens, layer_id)

            new_hidden_list.append(hidden)

        initial_hiddens = new_hidden_list

    print(calibrator.exit_params())
    calibrator.calibrate_K(hidden_states_list)
    calibrator.calibrate_threshold(hidden_states_list)
    print(calibrator.exit_params())


# test code
if __name__ == '__main__':
    
    import os, sys
    from pool import LayerwiseHiddenPool
    from accelerate.hooks import remove_hook_from_module

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../transformers/src"))
    print(project_root)
    sys.path.insert(0, project_root) 

    from transformers import AutoModelForCausalLM, AutoTokenizer
    model_name = '/datasets/ai/llama3/hub/models--meta-llama--Llama-3.2-1B/snapshots/4e20de362430cd3b72f300e6b0f18e50e7166e08'
    model_name='/datasets/ai/llama3/hub/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659'
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

    # Init components
    pool = LayerwiseHiddenPool()
    layer_manager = LayerManager(model, tok, pool, device="cuda")
    controller = PipelineController(
        layer_manager=layer_manager,
        pool=pool,
        min_batch_size=120,
        max_batches_per_layer=2
    )
    calibrator = Calibration(layer_manager)

    # Without this, no early exit
    layer_manager.set_calibrator(calibrator)

    # Calibrating
    calibrate(calibrator)

    
    # Process early exit based on calibrator's exit conditions
    if not layer_manager.is_layer_active(0):
        layer_manager.switch_active_layers(start_layer=0)

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
    ] * 50
    
    hidden_states_list = []
    for i in range(len(true_false_items)):
        prompt = true_false_items[i][0]
        input_ids = tok(prompt, return_tensors="pt").to("cuda")["input_ids"]
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