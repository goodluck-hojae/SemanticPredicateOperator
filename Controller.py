import time

class PipelineController:
    def __init__(
        self,
        layer_manager,
        pool,
        min_batch_size=2,
        max_batches_per_layer=2,
        next_fill_threshold=8,
    ):
        self.layer_manager = layer_manager
        self.pool = pool
        self.current_layer = 0
        self.min_batch_size = min_batch_size
        self.max_batches_per_layer = max_batches_per_layer
        self.next_fill_threshold = next_fill_threshold
        self.layer_batch_counter = 0

    def step(self):
        # Fetch
        hidden_batch, remaining = self.pool.fetch(self.current_layer, batch_size=self.min_batch_size)
        if hidden_batch is None:
            print(f"[Layer {self.current_layer}] No data, waiting...")
            return self._backtrack()

        # Forward
        hiddens = self.layer_manager.execute_hiddens(hidden_batch)
        for h in hiddens:
            if h.exit_layer is None:
                self.pool.store(h, h.layer_id)

        self.layer_batch_counter += 1
        print(f"[Layer {self.current_layer}] Processed {len(hidden_batch)} samples (remaining {remaining})")

        if self.layer_batch_counter >= self.max_batches_per_layer or remaining == 0:
            self.layer_batch_counter = 0
            next_layer = self.current_layer + 1

            # Final layer reached
            if next_layer >= self.layer_manager.num_layers():
                return self._backtrack()

            # Check if next layer pool has enough data
            next_count = self.pool.get_size(next_layer)
            if next_count < self.next_fill_threshold:
                print(f"Layer {next_layer} has only {next_count} samples (< {self.next_fill_threshold}) → stay on layer {self.current_layer}")
            else:
                print(f"Layer {next_layer} now has {next_count} samples → move to layer {next_layer}")
                self.current_layer = next_layer

                # Load next layers to GPU
                if self.current_layer >= self.layer_manager.top_layer:
                    print(f"Swapping active layer block: loading from layer {self.current_layer}")
                    self.layer_manager.switch_active_layers(start_layer=self.current_layer)
        return True

    def _backtrack(self):
        total_remaining = sum(len(v) for v in self.pool.hidden_states.values())
        if total_remaining == 0:
            print("All layers empty — pipeline fully complete.")
            return False

        # Find the earliest layer that still has pending hidden states
        for layer_id in range(self.layer_manager.num_layers()):
            count = self.pool.get_size(layer_id)
            if count > 0:
                print(f"Backtracking: layer {layer_id} still has {count} samples → returning to it")
                self.current_layer = layer_id

                # Ensure the correct GPU block is loaded
                print(f"Loading block containing layer {layer_id}")
                self.layer_manager.switch_active_layers(start_layer=layer_id)
                return True
        return False
