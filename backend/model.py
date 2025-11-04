import os, sys
import time
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../transformers/src"))
print(project_root)
sys.path.insert(0, project_root) 

import torch 
import torch.nn.functional as F
from transformers.cache_utils import DynamicCache, PreciseKV

from hidden import Hidden

class LayerManager:
    def __init__(self, model, tokenizer, pool, device='cuda'):
        self.active_layers = []
        self.device = device
        self.tokenizer = tokenizer
        self.pool = pool
        self.config = model.config
        self._init(model)
        self.kv_cache_dict = {} # id -> precise KV

    @torch.no_grad()
    def _init(self, model):
        self.embed_tokens = model.model.embed_tokens
        self.norm = model.model.norm
        self.rotary_emb = model.model.rotary_emb
        self.model_layers = model.model.layers
        self.lm_head = model.lm_head
        self.layer_capacity = self._layer_capacity(self.model_layers[0])
        self.calibrator = None
        self.kv_cache_capacity(500)
        print()

        # Check available VRAM and decide how many layers it holds
        for layer in self.model_layers[:self.layer_capacity]:
            self.active_layers.append(layer.to(self.device, non_blocking=True))

        self.top_layer = self.layer_capacity
    
    def set_calibrator(self, calibrator):
        self.calibrator = calibrator
        
    def is_layer_active(self, layer_id):
        end = self.top_layer
        start = max(0, self.top_layer - self.layer_capacity) #+1
        if start <= layer_id and layer_id < end:
            return True
        return False
         
    # Let's say 70%
    def _layer_capacity(self, layer, device='cuda'):
        # Available VRAM
        props = torch.cuda.get_device_properties(device)
        total_mem = props.total_memory

        # Layer size
        param_bytes = sum(p.numel() * p.element_size() for p in layer.parameters())
        buffer_bytes = sum(b.numel() * b.element_size() for b in layer.buffers())
        layer_bytes = param_bytes + buffer_bytes
        # print(free_mem, layer_bytes)
        allocated_layers = min(int((total_mem * 0.7) / layer_bytes), self.num_layers())
        print(f'{allocated_layers} layers will be allocated in the GPU memory')
        return allocated_layers

    # Let's say 20%
    # Memory bound for KV caches
    def kv_cache_capacity(self, avg_seq_len, device='cuda'):
        # Available VRAM
        props = torch.cuda.get_device_properties(device)
        total_mem = props.total_memory

        single_kv_cache_size = self.estimate_kv_size(avg_seq_len, dtype=torch.float16)

        allocated_layers = int((total_mem * 0.2) / single_kv_cache_size)
        print(f'{allocated_layers} samples as Block allocated in the GPU memory')
        return allocated_layers
    

    # Compute bound for a given GPU
    def compute_bound(self, avg_seq_len):
        import pynvml

        pynvml.nvmlInit()


        prompt = "test" * avg_seq_len
        input_ids = self.tokenizer(prompt, return_tensors="pt").to("cuda")['input_ids']
        
        hidden_states, position_ids, position_embeddings = layer_manager.process_input_tokens(input_ids, prompt)

        h=Hidden(id=0, states=hidden_states, prompt=prompt, pos_ids=position_ids, pos_emb=position_embeddings)
        print('estimating..')
        for idx in range(80):
            self.forward_layer(h, False)
            h.layer_id = 1
            
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)  # GPU 0
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            
            print(f"{idx} GPU Utilization: {util.gpu}%")
            print(f"{idx} Memory Utilization: {util.memory}%")
        print('estimating done')

        pynvml.nvmlShutdown()

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


    # TODO: Consider hiddenstates communication between CPU & GPU (prefetch logic)
    @torch.no_grad()
    def execute_hiddens(self, hiddens, early_exit=True):
        streams = [torch.cuda.Stream() for _ in hiddens]

        for h, s in zip(hiddens, streams):
            h.states = h.states.to(self.device, non_blocking=True)

            # Stream
            if s is not None:
                with torch.cuda.stream(s):
                    self.forward_layer(h, early_exit)
            else:
                self.forward_layer(h, early_exit)

        for s in streams:
            s.synchronize()

        return hiddens
    
    
    @torch.no_grad()
    def forward_layer(self, hidden, early_exit=True):
        # TODO I need A seq len so that i'd allocate block size as many as it needs. Specify hidden cache position, shape is [1, seq_len] along with pos_ids
        precise_cache = None
        if hidden.cache_position is not None:
            precise_cache = PreciseKV(
                num_layers=self.config.num_hidden_layers,
                num_heads=getattr(self.config, "num_key_value_heads", self.config.num_attention_heads),
                head_dim=self.config.hidden_size // self.config.num_attention_heads,
                block_size=len(hidden.cache_position),
                device=self.device)
            self.kv_cache_dict[hidden.prompt] = precise_cache # TODO Fix cache dict id

        next_hidden_states = self.model_layers[hidden.layer_id](
            hidden.states,
            attention_mask=None,
            position_ids=hidden.pos_ids,
            past_key_value=precise_cache,
            use_cache=True,
            cache_position=hidden.cache_position, # Save as much as I need 
            position_embeddings=hidden.pos_emb,
        )[0] # ???
        hidden.layer_id += 1
        hidden.states = next_hidden_states

        # Exit at last layer 
        if hidden.layer_id == self.num_layers():
            top_tokens = self.top_tokens(hidden.states)
            hidden.prediction_token = top_tokens[0]
            hidden.exit_layer = hidden.layer_id
            return True
        return self.check_exit_conditions(hidden) if early_exit else False
        

    @torch.no_grad()
    def check_exit_conditions(self, hidden):
        if self.calibrator is None:
            return False
        calibrator = self.calibrator
        exit = calibrator.check_exit(hidden.states)
        if exit:
            top_tokens = self.top_tokens(hidden.states)
            hidden.prediction_token = top_tokens[0]
            hidden.exit_layer = hidden.layer_id
            return True
        return False


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
        

    @torch.no_grad()
    def top_tokens(self, hidden_state):
        logits = self.lm_head(self.norm(hidden_state)[:, -1, :])
        topK = torch.topk(logits[0], k=10)
        top_tokens = [self.tokenizer.decode([tok]) for tok in topK.indices.tolist()]
        return top_tokens
    

    @torch.no_grad()
    def estimate_kv_size(self, seq_len, num_layers=None, dtype=torch.float16):
        hidden = self.config.hidden_size
        q_heads = self.config.num_attention_heads
        kv_heads = getattr(self.config, "num_key_value_heads", q_heads)
        head_dim = hidden // q_heads
        elems = 2 * kv_heads * seq_len * head_dim 
        bytes_per_elem = torch.tensor([], dtype=dtype).element_size()
        if num_layers is None:
            num_layers = self.num_layers()
        return elems * bytes_per_elem * num_layers


    def estimate_flops(self, seq_len, dtype=torch.float16):
        device = "cuda"
    
        num_layers = self.num_layers()
        hidden_size = self.config.hidden_size


        prompt = "test" * seq_len
        input_ids = self.tokenizer(prompt, return_tensors="pt").to("cuda")['input_ids']

        hidden_states, position_ids, position_embeddings = layer_manager.process_input_tokens(input_ids, prompt)

        h=Hidden(id=0, states=hidden_states, prompt=prompt, pos_ids=position_ids, pos_emb=position_embeddings)
        print('estimating..')

        n =  500
        # Warmup
        with torch.no_grad():
            for _ in range(10):
                self.forward_layer(h, False)
                h.layer_id = 1

        # Timed runs 
        streams = [torch.cuda.Stream() for _ in range(8)]
        torch.cuda.synchronize()
        t0 = time.time()
        start = time.time()
        with torch.no_grad():
            for i in range(n):
                s = streams[i % len(streams)]
                with torch.cuda.stream(s):
                    self.forward_layer(h, False)
                h.layer_id = 1
        torch.cuda.synchronize()

        latency = (time.time() - t0) / n
        tput = n / latency  # sequences/sec

        print(f"bs={n:2d}  latency={latency:.4f}s  seq/s={tput:.1f}")

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
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, 
                                                device_map={
                                                        "model.embed_tokens": "cuda",
                                                        **{f"model.layers.{i}": "cpu" for i in range(80)},  # all but last 2
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
    layer_manager.switch_active_layers(0)
    layer_manager.estimate_kv_size(500, 16)
    # layer_manager.compute_bound(500)
    layer_manager.estimate_flops(500)
    