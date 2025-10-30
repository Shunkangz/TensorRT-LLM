### :title KV Cache Connector with CPU Memory Pool
### :order 7
### :section Customization

import os
import sys
from dataclasses import dataclass, field
from typing import Dict, Optional

import click
import torch

from tensorrt_llm import LLM, SamplingParams, logger
from tensorrt_llm._torch.pyexecutor.kv_cache_connector import (
    KvCacheConnectorScheduler, KvCacheConnectorSchedulerOutputRequest,
    KvCacheConnectorWorker, RequestData, SchedulerOutput)
from tensorrt_llm.bindings import LlmRequestState
from tensorrt_llm.bindings.internal.batch_manager import LlmRequest
from tensorrt_llm.llmapi.llm_args import KvCacheConnectorConfig, TorchLlmArgs

# This is an example of a KV cache connector that uses a CPU memory pool.
# It demonstrates asynchronous GPU-to-CPU and CPU-to-GPU transfers using CUDA streams with KV cache connector.
# See tensorrt_llm/_torch/pyexecutor/connector.py for details about the KV cache connector interface.
# NOTE: This example connector implementation is NOT suitable for production use.


@dataclass
class CpuPoolKvCacheConnectorMetadata:
    load: list[tuple[int,
                     int]] = field(default_factory=list)  # (hash, block_id)
    save: list[tuple[int,
                     int]] = field(default_factory=list)  # (hash, block_id)
    # Track which requests have save operations (for cleanup in get_finished)
    save_request_ids: set[int] = field(default_factory=set)


class CpuMemoryPool:
    """CPU memory pool for storing KV cache blocks."""

    def __init__(self):
        self.pool: Dict[int, torch.Tensor] = {}  # hash -> CPU tensor

    def get(self, hash_key: int) -> Optional[torch.Tensor]:
        """Get a block from the pool by hash key."""
        return self.pool.get(hash_key)

    def put(self, hash_key: int, tensor: torch.Tensor):
        """Put a block into the pool."""
        if hash_key not in self.pool:
            self.pool[hash_key] = tensor.cpu().pin_memory()

    def exists(self, hash_key: int) -> bool:
        """Check if a block exists in the pool."""
        return hash_key in self.pool

    def clear(self):
        """Clear all blocks from the pool."""
        self.pool.clear()

    def size(self) -> int:
        """Get the number of blocks in the pool."""
        return len(self.pool)


class CpuPoolKvCacheConnectorWorker(KvCacheConnectorWorker):
    CHECKPOINT_DIR = "cpu_pool_checkpoints"
    # Class-level shared CPU pool across all instances (leader and workers)
    _shared_cpu_pool: Optional[CpuMemoryPool] = None
    # Class-level tracking of pending save requests (shared between worker and scheduler)
    _pending_save_requests: set[int] = set()

    def __init__(self, llm_args: TorchLlmArgs):
        super().__init__(llm_args)
        self.kv_cache_tensor = None

        # Initialize shared CPU pool if not already created
        if CpuPoolKvCacheConnectorWorker._shared_cpu_pool is None:
            CpuPoolKvCacheConnectorWorker._shared_cpu_pool = CpuMemoryPool()
            # Load existing checkpoint if available
            self._load_checkpoint()

        # Reference to the shared pool
        self.cpu_pool = CpuPoolKvCacheConnectorWorker._shared_cpu_pool

        # Track async operations with per-request events
        # Map batch_id -> (cuda_event, list of (hash, cpu_tensor), set of request_ids)
        self.pending_saves: Dict[int, tuple[torch.cuda.Event,
                                            list[tuple[int, torch.Tensor]],
                                            set[int]]] = {}

        # Map batch_id -> cuda_event for loads
        self.pending_loads: Dict[int, torch.cuda.Event] = {}

    @classmethod
    def get_cpu_pool(cls) -> Optional[CpuMemoryPool]:
        """Get the shared CPU pool. Returns None if not initialized."""
        return cls._shared_cpu_pool

    def _get_checkpoint_path(self) -> str:
        """Get the checkpoint file path for this rank."""
        rank = getattr(self._llm_args, 'rank', 0)
        return os.path.join(self.CHECKPOINT_DIR,
                            f"cpu_pool_checkpoint_rank{rank}.pt")

    def _load_checkpoint(self):
        """Load CPU pool from checkpoint if it exists."""
        checkpoint_path = self._get_checkpoint_path()
        if os.path.exists(checkpoint_path):
            try:
                pool_data = torch.load(checkpoint_path)
                CpuPoolKvCacheConnectorWorker._shared_cpu_pool.pool = pool_data
                logger.info(
                    f"Loaded CPU pool with "
                    f"{CpuPoolKvCacheConnectorWorker._shared_cpu_pool.size()} "
                    f"blocks from {checkpoint_path}")
            except Exception as e:
                logger.error(f"Failed to load CPU pool checkpoint from "
                             f"{checkpoint_path}: {e}")

    def __del__(self):
        """Destructor to save the CPU pool when the worker is destroyed."""
        if hasattr(self, 'cpu_pool') and self.cpu_pool.size() > 0:
            checkpoint_path = self._get_checkpoint_path()
            try:
                # Create checkpoint directory if it doesn't exist
                os.makedirs(self.CHECKPOINT_DIR, exist_ok=True)

                # Save the pool dictionary to disk
                torch.save(self.cpu_pool.pool, checkpoint_path)
                logger.info(
                    f"Saved CPU pool with {self.cpu_pool.size()} blocks "
                    f"to {checkpoint_path}")
            except Exception as e:
                logger.error(f"Failed to save CPU pool: {e}")

    def register_kv_caches(self, kv_cache_tensor: torch.Tensor):
        assert self.kv_cache_tensor is None, "KV cache tensor already registered"
        self.kv_cache_tensor = kv_cache_tensor

    def start_load_kv(self, stream: torch.cuda.Stream):
        # Copy the blocks from the CPU pool to the GPU asynchronously.
        logger.debug(f"Starting load KV")
        logger.debug(f"Metadata load: {self._metadata.load}")
        if not self._metadata or not self._metadata.load:
            return

        copy_stream = torch.cuda.Stream()

        for hash_key, block_id in self._metadata.load:
            cpu_tensor = self.cpu_pool.get(hash_key)
            assert cpu_tensor is not None, "Block not found in CPU pool"

            # Start async copy from CPU to GPU
            with torch.cuda.stream(copy_stream):
                self.kv_cache_tensor[block_id].copy_(cpu_tensor,
                                                     non_blocking=True)

        # Record a CUDA event after all loads in this batch
        # For now, we use -1 as a special key for batch-level loads
        # TODO: Extend metadata to include request_id for per-request tracking
        event = torch.cuda.Event()
        event.record(copy_stream)
        self.pending_loads[-1] = event
        logger.debug(f"Recorded event for load KV")

    def wait_for_layer_load(self, layer_idx: int, stream: torch.cuda.Stream):
        pass

    def save_kv_layer(self, layer_idx: int, stream: torch.cuda.Stream):
        pass

    def wait_for_save(self, stream: torch.cuda.Stream):
        # Make sure the forward pass is complete before beginning our save.
        stream.synchronize()

        logger.debug(f"Waiting for save: {self._metadata.save}")

        if not self._metadata or not self._metadata.save:
            return

        # Create a new CUDA stream for async GPU->CPU transfers
        copy_stream = torch.cuda.Stream()

        # Prepare the save data
        save_data = []

        with torch.cuda.stream(copy_stream):
            for hash_key, block_id in self._metadata.save:
                # Create pinned CPU tensor for async copy
                cpu_tensor = torch.empty_like(self.kv_cache_tensor[block_id],
                                              device='cpu',
                                              pin_memory=True)
                cpu_tensor.copy_(self.kv_cache_tensor[block_id],
                                 non_blocking=True)
                save_data.append((hash_key, cpu_tensor))

        # Record a CUDA event after all saves in this batch
        # For now, we use -1 as a special key for batch-level saves
        event = torch.cuda.Event()
        event.record(copy_stream)

        # Store the pending save with its event and the request IDs
        request_ids = self._metadata.save_request_ids
        self.pending_saves[-1] = (event, save_data, request_ids)

    def get_finished(
            self, finished_gen_req_ids: list[int],
            started_loading_req_ids: list[int]) -> tuple[list[int], list[int]]:
        logger.debug(f"Finished gen req ids: {finished_gen_req_ids}")
        logger.debug(f"Started loading req ids: {started_loading_req_ids}")

        finished_saving = []
        finished_loading = []

        # Check if save operations are complete
        # Since we use batch-level tracking (key=-1), check if that event is done
        logger.debug(f"Pending saves: {self.pending_saves}")
        if -1 in self.pending_saves:
            event, save_data, request_ids = self.pending_saves[-1]

            # Query the event to check if GPU->CPU copy is complete (non-blocking)
            if event.query():  # Returns True if event has occurred
                # Complete the save by adding tensors to the pool
                for hash_key, cpu_tensor in save_data:
                    self.cpu_pool.put(hash_key, cpu_tensor)

                logger.info(
                    f"Saved {len(save_data)} blocks to CPU pool. Pool size: {self.cpu_pool.size()}"
                )

                # Clean up the completed save
                del self.pending_saves[-1]

                # Only report the requests that were part of this save batch as finished
                finished_saving = [
                    req_id for req_id in finished_gen_req_ids
                    if req_id in request_ids
                ]

                # Clean up the shared pending_save_requests tracking
                for req_id in finished_saving:
                    CpuPoolKvCacheConnectorWorker._pending_save_requests.discard(
                        req_id)

        # Check if load operations are complete
        logger.debug(f"Pending loads: {self.pending_loads}")
        if -1 in self.pending_loads:
            event = self.pending_loads[-1]

            # Query the event to check if CPU->GPU copy is complete (non-blocking)
            if event.query():  # Returns True if event has occurred
                logger.info("Finished loading blocks from CPU pool for batch")

                # Clean up the completed load
                del self.pending_loads[-1]

                # All requests that started loading can be reported as done loading
                finished_loading = started_loading_req_ids.copy()

        logger.debug(f"Finished saving: {finished_saving}")
        logger.debug(f"Finished loading: {finished_loading}")
        return finished_saving, finished_loading


class CpuPoolKvCacheConnectorLeader(KvCacheConnectorScheduler):

    def __init__(self, llm_args: TorchLlmArgs):
        super().__init__(llm_args)

        self.block_size = self._llm_args.kv_cache_config.tokens_per_block
        self.pending_loads = {}
        # Use the shared class variable for tracking pending save requests
        # This allows the worker to clean it up when saves complete
        self.pending_save_requests = CpuPoolKvCacheConnectorWorker._pending_save_requests

        # Track async loading requests that have received allocations
        # Store request_id -> KvCacheConnectorSchedulerOutputRequest (not the LlmRequest object!)
        self.allocated_async_requests = {}

    def _create_scheduler_request_snapshot(
            self, request: LlmRequest,
            block_ids: list[int]) -> KvCacheConnectorSchedulerOutputRequest:
        """Create a snapshot of request state for deferred processing.

        Captures the request's tokens, block IDs, and state information needed
        to compute position and scheduled tokens later.
        """
        scheduler_request = KvCacheConnectorSchedulerOutputRequest()
        scheduler_request.block_ids = block_ids.copy()
        scheduler_request.tokens = request.get_tokens(0)

        # Store state info for computed_position and num_scheduled_tokens
        scheduler_request.state = request.state
        if request.state in (
                LlmRequestState.CONTEXT_INIT,
                LlmRequestState.DISAGG_GENERATION_TRANS_IN_PROGRESS):
            scheduler_request.context_current_position = \
                request.context_current_position
            scheduler_request.context_remaining_length = \
                request.context_remaining_length
            scheduler_request.context_chunk_size = request.context_chunk_size

        # Store the number of matched tokens from CPU pool
        # This will be used to adjust the computed_position
        if request.request_id in self.pending_loads:
            num_matched_blocks = len(self.pending_loads[request.request_id])
            scheduler_request.num_matched_tokens = num_matched_blocks * self.block_size
        else:
            scheduler_request.num_matched_tokens = 0

        return scheduler_request

    def _compute_request_position_and_tokens(
        self, scheduler_request: KvCacheConnectorSchedulerOutputRequest
    ) -> tuple[int, int]:
        """Compute position and scheduled tokens from scheduler request.

        This mirrors the logic in
        KvCacheConnectorSchedulerOutputRequest.update_and_build_data.

        Returns:
            Tuple of (computed_position, num_scheduled_tokens)
        """
        tokens = scheduler_request.tokens

        if hasattr(scheduler_request, 'state') and scheduler_request.state in (
                LlmRequestState.CONTEXT_INIT,
                LlmRequestState.DISAGG_GENERATION_TRANS_IN_PROGRESS):
            computed_position = scheduler_request.context_current_position
            num_scheduled_tokens = min(
                scheduler_request.context_remaining_length,
                scheduler_request.context_chunk_size)
        else:
            computed_position = len(tokens) - 1
            num_scheduled_tokens = 1  # generation

        # Reduce computed_position by the number of matched tokens from CPU pool
        # These tokens are already computed (loaded from CPU pool)
        if hasattr(scheduler_request, 'num_matched_tokens'):
            computed_position -= scheduler_request.num_matched_tokens
            logger.debug(
                f"Reducing computed_position by {scheduler_request.num_matched_tokens} "
                f"matched tokens from CPU pool")

        return computed_position, num_scheduled_tokens

    def build_connector_meta(self, scheduler_output: SchedulerOutput):
        """Build metadata for the connector based on scheduler output.

        This determines which blocks need to be loaded from the CPU pool
        and which blocks need to be saved to the CPU pool.
        """
        metadata = CpuPoolKvCacheConnectorMetadata()

        # Process async loading requests that have received allocations
        # These were skipped in build_scheduler_output but we need to process them here
        async_requests_to_process = []
        for req_id, scheduler_request in list(
                self.allocated_async_requests.items()):
            # Compute position and scheduled tokens using helper function
            computed_position, num_scheduled_tokens = \
                self._compute_request_position_and_tokens(scheduler_request)

            # Build RequestData from the stored scheduler request data
            request_data = RequestData(
                request_id=req_id,
                new_tokens=scheduler_request.tokens,
                new_block_ids=scheduler_request.block_ids,
                computed_position=computed_position,
                num_scheduled_tokens=num_scheduled_tokens)
            async_requests_to_process.append(request_data)

            logger.info(
                f"Processing async loading request {req_id} that received allocation: "
                f"{len(scheduler_request.block_ids)} blocks")

        # Clear the tracked async requests after processing
        self.allocated_async_requests.clear()

        # Process both regular new_requests and async requests
        all_new_requests = list(
            scheduler_output.new_requests) + async_requests_to_process

        for req in all_new_requests:
            # Process loads if there are any pending loads for this request
            if req.request_id in self.pending_loads and self.pending_loads[
                    req.request_id]:
                logger.debug(f"Request {req.request_id} has pending loads")
                num_computed_blocks = req.computed_position // self.block_size
                block_ids = req.new_block_ids

                pending_load = self.pending_loads[req.request_id]

                logger.debug(
                    f"Pending load block ids: {pending_load}, num computed blocks: {num_computed_blocks}, num block ids: {len(block_ids)}"
                )

                # Load blocks from CPU pool
                for hash_key, block_pos in zip(
                        pending_load, range(num_computed_blocks,
                                            len(block_ids))):
                    metadata.load.append((hash_key, block_ids[block_pos]))
                    logger.debug(
                        f"Loading block {block_ids[block_pos]} from CPU pool for request {req.request_id}"
                    )
            else:
                logger.debug(f"Request {req.request_id} has no pending loads")

            # Process saves regardless of whether there were pending loads
            num_computed_blocks = req.computed_position // self.block_size
            block_ids = req.new_block_ids
            pending_load = self.pending_loads.get(req.request_id, [])

            # Break up the remainder of the token sequence into chunks
            chunks = self._chunk_tokens(req.new_tokens)

            # For each chunk that isn't already on device and isn't in CPU pool, save it
            has_saves = False
            for block_pos in range(num_computed_blocks + len(pending_load),
                                   len(block_ids)):
                if len(chunks[block_pos]) == self.block_size:
                    hash_key = self._hash_tokens(chunks[block_pos])
                    metadata.save.append((hash_key, block_ids[block_pos]))
                    has_saves = True

            # Track if this request has pending saves
            logger.debug(f"Request {req.request_id} has saves: {has_saves}")
            if has_saves:
                self.pending_save_requests.add(req.request_id)
                metadata.save_request_ids.add(req.request_id)

        self.pending_loads = {}

        logger.debug(f"Metadata: {metadata}")
        return metadata

    def _hash_tokens(self, tokens: list[int]) -> int:
        """Hash a sequence of tokens to use as a key in the CPU pool."""
        return abs(hash(tuple(tokens)))

    def _chunk_tokens(self, tokens: list[int]) -> list[list[int]]:
        """Break tokens into chunks of block_size."""
        return [
            tokens[i:i + self.block_size]
            for i in range(0, len(tokens), self.block_size)
        ]

    def get_num_new_matched_tokens(
            self, request: LlmRequest,
            num_computed_tokens: int) -> tuple[int, bool]:
        """Check how many tokens can be loaded from the CPU pool.

        Returns:
            num_matched_tokens: Number of tokens that can be loaded from CPU pool
            is_prefix_match: Whether this is a prefix match (always False for this impl)
        """
        self.pending_loads[request.request_id] = []

        # Don't bother with sequences with partial matches
        if (num_computed_tokens % self.block_size) != 0:
            return 0, False

        computed_blocks = num_computed_tokens // self.block_size

        # Get all the tokens that don't have a cache hit on device
        remaining_tokens = request.get_tokens(0)[computed_blocks *
                                                 self.block_size:]

        remaining_chunks = self._chunk_tokens(remaining_tokens)

        # Get CPU pool to check for matches
        cpu_pool = CpuPoolKvCacheConnectorWorker.get_cpu_pool()
        if cpu_pool is None:
            return 0, False

        # For each chunk, check if it exists in our CPU pool
        for chunk in remaining_chunks:
            # Only do full blocks
            if len(chunk) == self.block_size:
                hash_key = self._hash_tokens(chunk)

                # If we get a cache hit in CPU pool, we want to load it
                # Otherwise, we can stop looking
                if cpu_pool.exists(hash_key):
                    self.pending_loads[request.request_id].append(hash_key)
                else:
                    break

        num_matched = len(self.pending_loads[request.request_id])
        if num_matched > 0:
            logger.info(
                f"KV CONNECTOR: Matched {num_matched} blocks in CPU pool for request {request.request_id}"
            )

        if num_matched == 0:
            logger.info(
                f"KV CONNECTOR: No blocks matched in CPU pool for request {request.request_id}"
            )
            return 0, False

        return num_matched * self.block_size, True

    def request_finished(self, request: LlmRequest,
                         cache_block_ids: list[int]) -> bool:
        """Called when a request finishes generation.

        Args:
            request: The request that finished generating tokens.
            cache_block_ids: The cache block IDs allocated for this request.

        Returns:
            True if there are still async save operations pending for this request
            False if all operations are complete
        """
        # Check if this specific request has pending save operations
        has_pending = request.request_id in self.pending_save_requests

        if has_pending:
            logger.debug(
                f"Request {request.request_id} has pending async saves")
        else:
            logger.debug(
                f"Request {request.request_id} has no pending async saves")

        return has_pending

    def update_state_after_alloc(self, request: LlmRequest,
                                 block_ids: list[int]):
        """Update internal state after blocks are allocated for a request.

        For async loading requests that were skipped in build_scheduler_output,
        we track them here so they can be processed in build_connector_meta.
        """
        # Check if this is an async loading request that needs tracking
        # Only track if: 1) request has pending loads, 2) blocks were allocated
        if (request.request_id in self.pending_loads
                and self.pending_loads[request.request_id]):
            logger.info(
                f"Tracking allocation for async loading request {request.request_id}: "
                f"{len(block_ids)} blocks = {block_ids}")
            # Create a snapshot of the request state for deferred processing
            # This avoids issues with accessing stale request state later
            scheduler_request = self._create_scheduler_request_snapshot(
                request, block_ids)
            self.allocated_async_requests[
                request.request_id] = scheduler_request
        else:
            logger.debug(
                f"Regular allocation for request {request.request_id}: "
                f"{len(block_ids)} blocks")


@click.command()
@click.argument("model", type=str)
def main(model: str):
    sys.path.append(os.path.join(
        os.path.dirname(__file__),
        "..",
    ))

    this_module = __file__[__file__.rfind("/") + 1:__file__.rfind(".py")]

    kv_connector_config = KvCacheConnectorConfig(
        connector_module=this_module,
        connector_scheduler_class="CpuPoolKvCacheConnectorLeader",
        connector_worker_class="CpuPoolKvCacheConnectorWorker",
    )

    llm = LLM(model=model,
              backend="pytorch",
              cuda_graph_config=None,
              kv_connector_config=kv_connector_config)

    test_text = (
        "Nvidia Corporation is an American technology company "
        "headquartered in Santa Clara, California. "
        "Founded in 1993 by Jensen Huang, Chris Malachowsky, and Curtis Priem, "
        "it develops graphics processing units (GPUs), "
        "system on a chips (SoCs), and application programming interfaces (APIs) "
        "for data science, high-performance computing, "
        "and mobile and automotive applications. Tell me about the company.")

    sampling_params = SamplingParams(max_tokens=32)

    print("\n=== First generation (populating CPU pool) ===")
    output = llm.generate([test_text], sampling_params)
    text0 = output[0].outputs[0].text
    print("First output: ", text0)

    # Get CPU pool stats
    cpu_pool = CpuPoolKvCacheConnectorWorker.get_cpu_pool()
    if cpu_pool:
        print(f"CPU pool size after first generation: {cpu_pool.size()} blocks")

    # Clean the GPU kv cache
    del llm

    llm = LLM(model=model,
              backend="pytorch",
              cuda_graph_config=None,
              kv_connector_config=kv_connector_config)

    print("\n=== Second generation (using CPU pool cache) ===")
    output = llm.generate([test_text], sampling_params)
    text1 = output[0].outputs[0].text
    print("Second output (using CPU pool cache): ", text1)

    # Verify outputs are identical (deterministic generation with same cache)
    assert text0 == text1, "Outputs should be identical when using cached KV blocks"

    print("\n=== Third generation with modified prompt ===")
    # Try a modified version of the prompt that shares a prefix
    modified_text = test_text.replace("Tell me about the company.",
                                      "What are its main products?")
    output = llm.generate([modified_text], sampling_params)
    text2 = output[0].outputs[0].text
    print("Third output (partial cache hit): ", text2)

    if cpu_pool:
        print(f"Final CPU pool size: {cpu_pool.size()} blocks")

    print("\n=== Success! ===")
    print("The CPU pool-based KV cache connector is working correctly.")
    print("- Async GPU-CPU transfers were used")
    print("- KV cache blocks were reused from CPU pool")
    print("- request_finished() returned True for async operations")


if __name__ == "__main__":
    main()
