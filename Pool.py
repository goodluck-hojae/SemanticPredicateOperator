import torch
from Hidden import Hidden


class LayerwiseHiddenPool:

    def __init__(self):
        self.hidden_states = {}

    def store(self, layer_id: int, hidden: Hidden):
        if layer_id not in self.hidden_states:
            self.hidden_states[layer_id] = []
        self.hidden_states[layer_id].append(hidden)

    def get(self, layer_id: int):
        return self.hidden_states.get(layer_id, [])

    def get_size(self, layer_id: int):
        return len(self.get(layer_id))
    
    def fetch(self, layer_id: int, batch_size=2, device='cuda'):
        hiddens = self.get(layer_id)
        if not hiddens or len(hiddens) == 0:
            return None, 0

        batch = hiddens[:self.batch_size]
        # TODO: prefetch logic
        for b in batch:
            b.to(device, non_blocking=True)
        del hiddens[:self.batch_size]
        remaining_count = max(0, len(hiddens))
        return batch, remaining_count

    def release(self, layer_id: int):
        if layer_id in self.hidden_states:
            del self.hidden_states[layer_id]

    def clear(self):
        self.hidden_states.clear()

    def __contains__(self, layer_id: int):
        return layer_id in self.hidden_states and len(self.hidden_states[layer_id]) > 0

    def __len__(self):
        return sum(len(v) for v in self.hidden_states.values())
