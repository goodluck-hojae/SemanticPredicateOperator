
class Hidden:
    def __init__(self, id, hidden, prompt, pos_emb):
        self.id = id
        self.prompt = prompt
        self.layer_idx = 0
        self.hidden = hidden
        self.pos_emb = pos_emb


    def __repr__(self):
        h_shape = getattr(self.hidden, "shape", None)
        p_shape = getattr(self.pos_emb[0], "shape", None)
        return (
            f"Hidden("
            f"id={self.id!r}, "
            f"prompt={self.prompt!r}, "
            f"layer_idx={self.layer_idx}, "
            f"hidden_shape={h_shape}, "
            f"pos_emb_shape={p_shape}"
            f")"
        )