import torch
from Hidden import Hidden


class LayerwiseHiddenPool:

    def __init__(self):
        self.hidden_states = {}

    def store(self, hidden: Hidden, layer_id:int=0):
        if layer_id not in self.hidden_states:
            self.hidden_states[layer_id] = []
        self.hidden_states[layer_id].append(hidden)
    
    def fetch(self, layer_id: int, batch_size=2, device='cuda'):
        hiddens = self.get(layer_id)
        if not hiddens or len(hiddens) == 0:
            return None, 0

        batch = hiddens[:batch_size]
        # TODO: prefetch logic
        for b in batch:
            b.to(device, non_blocking=True)
        del hiddens[:batch_size]
        remaining_count = max(0, len(hiddens))
        return batch, remaining_count

    def get(self, layer_id: int):
        return self.hidden_states.get(layer_id, [])

    def get_size(self, layer_id: int):
        return len(self.get(layer_id))

    def release(self, hidden):
        self.hidden_states[hidden.layer_id].remove(hidden)

    def clear(self):
        self.hidden_states.clear()
    
    def __contains__(self, layer_id: int):
        return layer_id in self.hidden_states and len(self.hidden_states[layer_id]) > 0

    def __len__(self):
        return sum(len(v) for v in self.hidden_states.values())


if __name__ == '__main__':

    import os, sys
    from Model import LayerManager
    from accelerate.hooks import remove_hook_from_module

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../transformers/src"))
    print(project_root)
    sys.path.insert(0, project_root) 

    from transformers import AutoModelForCausalLM, AutoTokenizer
    model_name = '/datasets/ai/llama3/hub/models--meta-llama--Llama-3.2-1B/snapshots/4e20de362430cd3b72f300e6b0f18e50e7166e08'
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, 
                                            device_map={
                                                    "model.embed_tokens": "cuda",
                                                    **{f"model.layers.{i}": "cpu" for i in range(16)},  # all but last 2
                                                    "model.norm": "cuda",
                                                    "lm_head": "cuda",
                                                })
    model.eval()
    for module in model.modules():
        remove_hook_from_module(module)


    pool = LayerwiseHiddenPool()
    layer_manager = LayerManager(model, pool)
    prompt = 'How are you?' * 100
    input_ids = tok(prompt, return_tensors="pt").to("cuda")['input_ids']
    

    for i in range(100):
        hidden_states, position_embeddings = layer_manager.process_input_tokens(input_ids, prompt)
        pool.store(layer_id=0, hidden=Hidden(id=i, hidden=hidden_states, prompt=prompt, pos_emb=position_embeddings))

    print('Terminating..')