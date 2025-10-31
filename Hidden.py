
class Hidden:
    def __init__(self, id, states, prompt, pos_ids, pos_emb):
        self.id = id
        self.prompt = prompt
        self.layer_id = 0
        self.states = states
        self.pos_ids = pos_ids
        self.pos_emb = pos_emb
        self.exit_layer = None
        self.prediction_token = None

    def to(self, device='cuda', non_blocking=True):
        self.states = self.states.to(device, non_blocking=non_blocking)
        self.pos_emb = [emb.to(device, non_blocking=non_blocking) for emb in self.pos_emb]
        return self
        
    def __repr__(self):
        h_shape = getattr(self.states, "shape", None)
        p_shape = getattr(self.pos_emb[0], "shape", None)
        return (
            f"Hidden("
            f"id={self.id!r}, "
            f"prompt={self.prompt!r}, "
            f"layer_id={self.layer_id}, "
            f"hidden_shape={h_shape}, "
            f"pos_emb_shape={p_shape}"
            f")"
        )