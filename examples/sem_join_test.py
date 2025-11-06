import os, sys
import time
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../transformers/src"))
print(project_root)
sys.path.insert(0, project_root) 

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
backend_path = os.path.join(project_root, "backend")
calibration_path = os.path.join(project_root, "calibration")
semantic_ops_path = os.path.join(project_root, "semantic_ops")
sys.path.extend([backend_path, calibration_path, semantic_ops_path])


if __name__ == '__main__':
    from prompt import Prompt, PromptConstructor
    from data import BlockPairLoader
    predicate_placeholder = "{a} is aligned with Document [{b}]?" + "test " *500 # Predicate statement
    prompt_constructor = PromptConstructor(placeholder=predicate_placeholder)
    
    tableA = []
    tableB = []
    for i in range(100):
        tableA.append(f'A-{i}')

    for i in range(100):
        tableB.append(f'B-{i}')
 
    
    from hidden import Hidden
    from pool import LayerwiseHiddenPool
    from model import LayerManager
    from scheduler import PipelineController

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from accelerate.hooks import remove_hook_from_module
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

    predicate_placeholder = "{a} is aligned with Document [{b}]?" + "test " *500 # Predicate statement
    prompt_constructor = PromptConstructor(placeholder=predicate_placeholder)
    
    tableA = []
    tableB = []
    for i in range(1000):
        tableA.append(f'A-{i}')

    for i in range(1000):
        tableB.append(f'B-{i}')

    avg_seq_len = 500
    kv_cache_limit = layer_manager.kv_cache_capacity(avg_seq_len)
    print('kv_cache_limit within available layers', kv_cache_limit)
    

    controller = PipelineController(
        layer_manager=layer_manager,
        pool=pool,
        min_batch_size=kv_cache_limit,
        max_batches_per_layer=1
    )
    

    bnlj = BlockPairLoader(int(kv_cache_limit ** 0.5), 3)

    i = 0
    pidx = 0
    for idx, block_pair in enumerate(bnlj.next(tableA, tableB)):
        blockA, blockB = block_pair
        for pageA in blockA.next_page():
            for pageB in blockB.next_page():
                for tupleA in pageA.prompt_list:
                    for tupleB in pageB.prompt_list:
                        prompt = prompt_constructor.construct_prompt(tupleA.statement, tupleB.statement)
                        input_ids = tok(prompt.statement, return_tensors="pt").to("cuda")['input_ids']
                        prompt.set_input_ids(input_ids)
                        hidden_states, poistion_ids, position_embeddings = layer_manager.process_input_tokens(prompt.input_ids)
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
                        i+=1
                
                print(f"Layer 0 pool initialized with {pool.get_size(0)} hiddens.\n")

                # page size 
                if pidx != 0 and pidx % 2 == 0:

                    print("=== Starting pipeline loop ===")
                    start_time = time.time()

                    step_count = 0
                    while True:
                        step_count += 1
                        cont = controller.step(early_exit=False)
                        if not cont:
                            break

                    print(f"=== Done in {time.time() - start_time:.2f}s after {step_count} steps ===")

                    for lid, hlist in pool.hidden_states.items():
                        print(f"Layer {lid}: {len(hlist)} remaining hiddens")
                pidx +=1