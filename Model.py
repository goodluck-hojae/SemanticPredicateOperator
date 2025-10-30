import torch 
from Hidden import Hidden

class LayerManager:
    def __init__(self, model, tokenizer, pool, device='cuda'):
        self.active_layers = []
        self.device = device
        self._init(model)
        self.tokenizer = tokenizer
        self.pool = pool


    def _init(self, model):
        self.embed_tokens = model.model.embed_tokens
        self.norm = model.model.norm
        self.rotary_emb = model.model.rotary_emb
        self.model_layers = model.model.layers
        self.lm_head = model.lm_head
        self.layer_capacity = self._layer_capacity(self.model_layers[0])

        # Check available VRAM and decide how many layers it holds
        for layer in self.model_layers[:self.layer_capacity]:
            self.active_layers.append(layer.to(self.device, non_blocking=True))

        self.top_layer = self.layer_capacity

    def _layer_capacity(self, layer, device='cuda'):
        # Check layer size & Check available GPU
        layer_capacity = 4
        return layer_capacity

    def num_layers(self):
        return len(self.model_layers)
    
    @torch.no_grad()
    def switch_active_layers(self, start_layer: int = None):
        with torch.no_grad():
            for layer in self.active_layers:
                layer.to('cpu')
            self.active_layers.clear()

        total_layers = len(self.model_layers)
        block = self.layer_capacity

        if start_layer is None:
            # next block
            start = self.top_layer if self.top_layer < total_layers else 0
        else:
            # specified block
            start = max(0, min(start_layer, total_layers - 1))

        end = min(start + block, total_layers)

        for layer in self.model_layers[start:end]:
            self.active_layers.append(layer.to(self.device, non_blocking=True))

        self.top_layer = end
        print(f">> Loaded layers {start}–{end-1} on {self.device}")


    @torch.no_grad()
    def execute_hiddens(self, hiddens, layer_id):
        streams = [torch.cuda.Stream() for _ in hiddens]

        for h, s in zip(hiddens, streams):
            self.forward_layer(h, layer_id, stream=s)

        for s in streams:
            s.synchronize()

        return hiddens
    

    @torch.no_grad()
    def forward_layer(self, hidden, layer_id, stream=None):
        layer = self.model_layers[layer_id]

        hidden.states = hidden.states.to(self.device, non_blocking=True)

        # Stream
        if stream is not None:
            with torch.cuda.stream(stream):
                hidden.states = layer(hidden.states)
                hidden.layer_idx += 1
        else:
            # Synchronous
            hidden.states = layer(hidden.states)
            hidden.layer_idx += 1


    @torch.no_grad()
    def forward_layer(self, hidden, layer_id, stream=None):
        next_hidden_states = self.model_layers[layer_id](
            hidden.states,
            attention_mask=None,
            position_ids=hidden.pos_ids,
            past_key_values=None,
            use_cache=True,
            cache_position=hidden.pos_ids,
            position_embeddings=hidden.pos_emb,
        )
        hidden.layer_idx += 1
        hidden.states = next_hidden_states
        logits =self.lm_head(self.norm(hidden.states)[:, -1, :])
        topK = torch.topk(logits[0], k=3)
        top_tokens = [self.tokenizer.decode([tok]) for tok in topK.indices.tolist()]
        if layer_id == len(self.model_layers)-1:
            print(layer_id, top_tokens)

    @torch.no_grad()
    def process_input_tokens(self, input_ids, prompt=None):
        inputs_embeds = self.embed_tokens(input_ids)
        cache_position = torch.arange(0, inputs_embeds.shape[1], device=inputs_embeds.device)
        position_ids = cache_position.unsqueeze(0)

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        return hidden_states, position_ids, position_embeddings


    def check_layer_device(self):
        for layer in self.model_layers:
            print(next(layer.parameters())[0].device)
        
# test code
if __name__ == '__main__':
    
    import os, sys
    from Pool import LayerwiseHiddenPool
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
    layer_manager = LayerManager(model, tok, pool)

    prompt = 'How are you?'
    input_ids = tok(prompt, return_tensors="pt").to("cuda")['input_ids']
    hidden_states, position_ids, position_embeddings = layer_manager.process_input_tokens(input_ids, prompt)
    pool.store(0, Hidden(id=0, states=hidden_states, prompt=prompt, pos_ids=position_ids, pos_emb=position_embeddings))
    print(tok, model)
    layer_manager.switch_active_layers()
    