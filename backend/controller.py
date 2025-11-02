import time

class PipelineController:
    def __init__(
        self,
        layer_manager,
        pool,
        min_batch_size,
        max_batches_per_layer
    ):
        self.layer_manager = layer_manager
        self.pool = pool
        self.min_batch_size = min_batch_size
        self.max_batches_per_layer = max_batches_per_layer 
        self.current_layer = 0
        self.layer_batch_counter = 0

    def step(self, early_exit=True):
        # Fetch
        hidden_batch, remaining = self.pool.fetch(self.current_layer, batch_size=self.min_batch_size)
        if hidden_batch is None:
            # If previous step decides to stay on the current layer and still there is no more data fetched, it moves to first range of layers again
            print(f"[Layer {self.current_layer}] No data, waiting...")
            return self._backtrack()

        # Forward
        hiddens = self.layer_manager.execute_hiddens(hidden_batch, early_exit)
        for h in hiddens:
            if h.exit_layer is None:
                self.pool.store(h, h.layer_id)
            elif h.layer_id == self.layer_manager.num_layers():
                print(f'{h.id} hidden states exited at the last layer with *-- {h.prediction_token} token --* at hidden.exit_layer {h.exit_layer}')
                del h
            else:
                print(f'{h.id} hidden states exited early with {h.prediction_token} at hidden.exit_layer {h.exit_layer}')    
                del h

        self.layer_batch_counter += 1
        print(f"[Layer {self.current_layer}] Processed {len(hidden_batch)} samples (remaining {remaining})")

        # Move to next layers after fetch data (max_batches_per_layer) times, else, It stays the current layer
        if self.layer_batch_counter >= self.max_batches_per_layer or remaining == 0:
            self.layer_batch_counter = 0
            next_layer = self.current_layer + 1

            # Final layer reached
            if next_layer >= self.layer_manager.num_layers():
                return self._backtrack()

            # Check if next layer pool has enough data
            next_count = self.pool.get_size(next_layer)
            if next_count < self.min_batch_size:
                print(f"Layer {next_layer} has only {next_count} samples (< {self.min_batch_size}) -> stay on layer {self.current_layer}")
            else:
                print(f"Layer {next_layer} has enough samples {next_count} (> {self.min_batch_size}) -> move to layer {next_layer}")
                self.current_layer = next_layer

                # Load next layers to GPU
                if self.current_layer >= self.layer_manager.top_layer and not self.layer_manager.is_layer_active(self.current_layer):
                    print(f"Swapping active layer block: loading from layer {self.current_layer}")
                    self.layer_manager.switch_active_layers(start_layer=self.current_layer)
        return True
            

    def _backtrack(self):
        total_remaining = len(self.pool)
        if total_remaining == 0:
            print("All layers empty — pipeline fully complete.")
            return False
            
        # backtrack to the first layer
        for layer_id in range(self.layer_manager.num_layers()):
            count = self.pool.get_size(layer_id)
            if count > 0:
                print(f"Backtracking: layer {layer_id} still has {count} samples -> loading its block.")
                self.current_layer = layer_id
                if not self.layer_manager.is_layer_active(self.current_layer):
                    print(f"Swapping active layer block: loading from layer in backtrack {self.current_layer}")
                    self.layer_manager.switch_active_layers(start_layer=layer_id)
                return True

        return False
    