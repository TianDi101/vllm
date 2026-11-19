# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared scheduler-side UMBP connector behavior."""

from __future__ import annotations

import contextlib
import logging
import os
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from concurrent.futures import wait as wait_futures
from dataclasses import dataclass, field, replace
from functools import lru_cache
from itertools import islice
from typing import Any, cast

from vllm.config import VllmConfig
from vllm.distributed.kv_events import BlockRemoved, BlockStored, KVCacheEvent
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorMetadata
from vllm.logger import init_logger
from vllm.v1.attention.backends.utils import NULL_BLOCK_ID
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_coordinator import (
    get_kv_cache_coordinator,
    replay_boundaries,
)
from vllm.v1.core.kv_cache_manager import KVCacheBlocks
from vllm.v1.core.kv_cache_utils import (
    BlockHash,
    KVCacheBlock,
    get_block_hash,
    make_block_hash_with_group_id,
    maybe_convert_block_hash,
    resolve_dcp_kv_block_size,
    resolve_kv_cache_block_sizes,
)
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheSpec,
    MambaSpec,
    SparseCacheRole,
    UniformTypeKVCacheSpecs,
    iter_layer_specs,
    mamba_tail_window,
)
from vllm.v1.kv_cache_spec_registry import KVCacheSpecRegistry
from vllm.v1.outputs import KVConnectorOutput
from vllm.v1.request import Request

from .data import (
    BlockIdentityCodec,
    BlockLoadBatch,
    BlockStoreBatch,
    BlockTransferPlan,
    LoadSpec,
    RankTopology,
    RequestTracker,
    StoreEventResult,
    UMBPConnectorMetadata,
    UMBPConnectorWorkerMetadata,
)
from .runtime import UMBPSchedulerHandle

logger = init_logger(__name__)


class _StepLog:
    """VLLM_UMBP_STEP_LOG=1: where the engine core's scheduling time goes.

    One `K3STEP` line per interval from the scheduler process: steps, tokens scheduled and
    how full the steps were, time inside schedule() / update_from_output() (core methods,
    timed by wrapping them), time inside this connector's lookups, and lookup latency.
    The worker's `K3WSTEP` line (worker.py) has the forward and idle time of rank 0.
    """

    INTERVAL_S = 10.0

    def __init__(self) -> None:
        self.t0 = time.monotonic()
        self.acc: dict[str, float] = {}
        self.lookup_lat: list[float] = []
        self._install_core_timers()

    def add(self, key: str, value: float = 1.0) -> None:
        self.acc[key] = self.acc.get(key, 0.0) + value

    def lookup_done(self, seconds: float) -> None:
        self.lookup_lat.append(seconds)

    def _install_core_timers(self) -> None:
        from vllm.v1.core.sched.scheduler import Scheduler

        first = not hasattr(Scheduler, "_umbp_step_log")
        Scheduler._umbp_step_log = self  # type: ignore[attr-defined]
        if not first:
            return
        schedule, update = Scheduler.schedule, Scheduler.update_from_output

        def timed_schedule(sched, *args, **kwargs):
            started = time.monotonic()
            try:
                return schedule(sched, *args, **kwargs)
            finally:
                Scheduler._umbp_step_log.add("sched_s", time.monotonic() - started)
                Scheduler._umbp_step_log.add("sched_n")

        def timed_update(sched, *args, **kwargs):
            started = time.monotonic()
            try:
                return update(sched, *args, **kwargs)
            finally:
                Scheduler._umbp_step_log.add("update_s", time.monotonic() - started)

        Scheduler.schedule = timed_schedule  # type: ignore[method-assign]
        Scheduler.update_from_output = timed_update  # type: ignore[method-assign]

    def step(self, tokens: int, requests: int, loads: int, budget: int) -> None:
        self.add("steps")
        self.add("tokens", tokens)
        self.add("reqs", requests)
        self.add("loads", loads)
        self.add("full" if tokens >= budget - 256 else ("part" if tokens else "empty"))
        now = time.monotonic()
        if now - self.t0 < self.INTERVAL_S:
            return
        acc, lat = self.acc, sorted(self.lookup_lat)
        g = lambda key: acc.get(key, 0.0)  # noqa: E731
        logger.info(
            "K3STEP dt=%.1f steps=%d tokens=%d full=%d part=%d empty=%d reqs=%d loads=%d "
            "sched_n=%d sched_s=%.3f update_s=%.3f bcm_s=%.3f gnnmt_n=%d gnnmt_none=%d "
            "gnnmt_s=%.3f await_s=%.3f await_n=%d lookups=%d lookup_ms_mean=%.1f "
            "lookup_ms_p90=%.1f lookup_ms_max=%.1f",
            now - self.t0, g("steps"), g("tokens"), g("full"), g("part"), g("empty"),
            g("reqs"), g("loads"), g("sched_n"), g("sched_s"), g("update_s"), g("bcm_s"),
            g("gnnmt_n"), g("gnnmt_none"), g("gnnmt_s"), g("await_s"), g("await_n"),
            len(lat), 1e3 * sum(lat) / len(lat) if lat else 0.0,
            1e3 * lat[int(0.9 * (len(lat) - 1))] if lat else 0.0,
            1e3 * lat[-1] if lat else 0.0,
        )
        self.t0 = now
        self.acc = {}
        self.lookup_lat = []


@lru_cache(maxsize=8192)
def _rank_keys(
    codec: BlockIdentityCodec,
    topology: RankTopology,
    block_hash: bytes,
    group_id: int,
) -> tuple[str, ...]:
    """Reuse immutable object identities, not runtime residency."""
    return codec.keys_for_topology(block_hash, topology, (group_id,))


def _unwrap_store_spec(spec: KVCacheSpec) -> KVCacheSpec:
    if isinstance(spec, UniformTypeKVCacheSpecs):
        return next(iter(spec.kv_cache_specs.values()))
    return spec


@dataclass
class _StoreEvent:
    plan_groups: tuple[Sequence[BlockTransferPlan], ...]
    result: StoreEventResult = field(default_factory=StoreEventResult)

    def iter_plans(self):
        for plans in self.plan_groups:
            yield from plans

    def iter_keys(self):
        for plans in self.plan_groups:
            if isinstance(plans, BlockStoreBatch):
                yield from plans.keys
            else:
                yield from (plan.key for plan in plans)

    def block_ids(self):
        for plans in self.plan_groups:
            if isinstance(plans, BlockStoreBatch):
                yield from plans.block_ids
            else:
                yield from (plan.block_id for plan in plans)


@dataclass(frozen=True)
class _LookupContext:
    request: Request
    local_tokens: int
    num_tokens: int
    hashes: tuple[bytes, ...]
    objects: dict[int, list[tuple[bytes, tuple[str, ...]]]]
    keys: list[str]


@dataclass
class _ArrivalPrefetch:
    """One request's arrival prefetch: lookup -> (core thread) plan -> SSD-to-DRAM copy."""

    request: Request
    hashes: tuple[bytes, ...]
    arrived: float
    local_tokens: int = 0
    context: Any = None
    lookup: Future | None = None
    copy: Future | None = None
    tokens: int = 0  # external tokens the copy covers; the credit it holds
    keys: int = 0
    started: float = 0.0
    dead: bool = False


@dataclass(frozen=True)
class _PendingLookup:
    context: _LookupContext
    future: Future[Sequence[bool]]


@dataclass
class _EventObject:
    serial: int
    plan: BlockTransferPlan | None
    published: bool = False


@dataclass(frozen=True)
class _EventLookup:
    objects: tuple[tuple[tuple[int, bytes], int], ...]
    future: Future[Sequence[bool]]


class _LookupBlockPool(BlockPool):
    """Request-scoped residency view for the core hit-window algorithms."""

    def __init__(self, hash_block_size: int) -> None:
        super().__init__(
            num_gpu_blocks=1,
            enable_caching=True,
            hash_block_size=hash_block_size,
            enable_kv_cache_events=False,
        )
        self.hits: list[set[bytes]] = []
        self.hit_blocks: dict[bytes, KVCacheBlock] = {}

    def get_cached_block(
        self, block_hash: bytes, kv_cache_group_ids: list[int]
    ) -> list[KVCacheBlock] | None:
        if any(block_hash not in self.hits[group] for group in kv_cache_group_ids):
            return None
        block = self.hit_blocks.get(block_hash)
        if block is None:
            block = KVCacheBlock(
                block_id=len(self.hit_blocks) + 1,
                block_hash=make_block_hash_with_group_id(
                    BlockHash(block_hash), kv_cache_group_ids[0]
                ),
            )
            self.hit_blocks[block_hash] = block
        return [block] * len(kv_cache_group_ids)

    def clear_lookup(self) -> None:
        self.hits = []
        self.hit_blocks.clear()


class UMBPStoreConnectorScheduler:
    """Own vLLM prefix matching while the runtime owns lookup transport.

    Without a runtime and codec the scheduler neither looks up nor stores
    until `bind_namespace` supplies them.
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        kv_cache_config: KVCacheConfig,
        runtime: UMBPSchedulerHandle | None,
        codec: BlockIdentityCodec | None,
        topology: RankTopology | None = None,
    ) -> None:
        self.block_size, resolved_hash_block_size = resolve_kv_cache_block_sizes(
            kv_cache_config, vllm_config
        )
        self.kv_cache_config = kv_cache_config
        dcp_size = vllm_config.parallel_config.decode_context_parallel_size
        self._dcp_size = dcp_size
        self._retention_interval = kv_cache_config.prefix_cache_retention_interval
        self.group_block_sizes = {
            group_id: resolve_dcp_kv_block_size(group.kv_cache_spec, dcp_size)
            for group_id, group in enumerate(kv_cache_config.kv_cache_groups)
        }
        # Mamba "align" block tables are not append-only: a position may hold
        # the live state the next step writes, or a speculative block core will
        # relocate, so positional stores would read or pin the wrong block.
        # Core hands off each committed boundary state instead
        # (_add_boundary_state_plans).
        self._boundary_handoff_groups = frozenset(
            group_id
            for group_id, group in enumerate(kv_cache_config.kv_cache_groups)
            if isinstance(group.kv_cache_spec, MambaSpec)
            and getattr(group.kv_cache_spec, "mamba_cache_mode", None) == "align"
        )
        # Tail-window states (VLLM_MAMBA_TAIL_WINDOW) are speculative: most are
        # never hit, so they and their attention tail pages go under the low
        # eviction class; every other boundary stays class 0.
        self._window_block_size = (
            min(self.group_block_sizes[g] for g in self._boundary_handoff_groups)
            if mamba_tail_window() is not None and self._boundary_handoff_groups
            else 0
        )
        self._full_block_size = max(self.group_block_sizes.values(), default=0)
        # Full-attention groups, including a group that wraps several attention
        # layer specs (target and draft layers sharing a block table).
        self._attention_tail_groups = frozenset(
            group_id
            for group_id in kv_cache_config.prefix_cacheable_group_ids
            if isinstance(
                _unwrap_store_spec(
                    kv_cache_config.kv_cache_groups[group_id].kv_cache_spec
                ),
                FullAttentionSpec,
            )
        )
        self.runtime = runtime
        self.enable_kv_cache_events = bool(
            vllm_config.kv_events_config
            and vllm_config.kv_events_config.enable_kv_cache_events
        )
        self.codec = codec
        self.topology = topology or RankTopology()
        transfer_config = vllm_config.kv_transfer_config
        assert transfer_config is not None
        extra = transfer_config.kv_connector_extra_config
        self.load_async = bool(extra.get("load_async", True))
        if not self.load_async and (
            len(kv_cache_config.kv_cache_groups) > 1
            or any(
                isinstance(spec, MambaSpec)
                or getattr(spec, "cache_role", None) == SparseCacheRole.INDEXER
                for group in kv_cache_config.kv_cache_groups
                for spec in iter_layer_specs(group.kv_cache_spec)
            )
        ):
            # Core accepts only request-level load failures for several cache
            # groups, and acts on them only while the request waits for an
            # asynchronous load. Mamba-style and sparse-attention indexer
            # layers also read restored state before any wait_for_layer_load.
            logger.warning(
                "UMBP loads asynchronously for models with several KV cache "
                "groups, Mamba-style or sparse-attention indexer layers; "
                "load_async=false is ignored"
            )
            self.load_async = True
        self.enable_lookup = bool(extra.get("enable_lookup", True))
        self.lookup_async = bool(extra.get("lookup_async", False))
        # An async lookup defers its request by a whole engine step, though most
        # answer within milliseconds: wait this long first so the request can
        # run now. The per-pass budget bounds how long one schedule() blocks.
        self._lookup_wait_s = float(extra.get("lookup_sync_wait_ms", 20)) / 1000
        self._lookup_pass_budget_s = (
            float(extra.get("lookup_sync_budget_ms", 50)) / 1000
        )
        self._lookup_budget_left_s = self._lookup_pass_budget_s
        self.save_decode_cache = bool(extra.get("save_decode_cache", False))
        # A kv_consumer only restores from the pool, as with other store
        # connectors; it never offloads its own KV.
        self.store_enabled = getattr(transfer_config, "kv_role", None) != "kv_consumer"
        # Request.block_hashes are computed at this size; keys index into them.
        self.hash_block_size = resolved_hash_block_size
        # VLLM_UMBP_HIT_LOG=1: one line per admitted request with its GPU (local)
        # hit, pool (external) hit and an infinite-cache bound, the longest prefix
        # of its block hashes that any earlier request produced (debug only).
        self._hit_log = os.environ.get("VLLM_UMBP_HIT_LOG", "0") == "1"
        self._step_log = (
            _StepLog() if os.environ.get("VLLM_UMBP_STEP_LOG", "0") == "1" else None
        )
        # Chain-ordered recency (VLLM_UMBP_CHAIN_TOUCH=1). A prefix is loadable only as an
        # unbroken run of pages from its start, but the pool evicts key by key in the order
        # keys were last touched, and recency hints went out shallowest page first: the
        # head of a prefix was the first page evicted, which stranded every page after it.
        # With this on, hints are sent deepest page first, and when a request finishes its
        # whole prompt is touched once more in that order (after its stores have landed),
        # so eviction takes a prefix from its deep end and what remains stays loadable.
        self._chain_touch = os.environ.get("VLLM_UMBP_CHAIN_TOUCH", "0") == "1"
        self._chain_touch_delay_s = float(
            os.environ.get("VLLM_UMBP_CHAIN_TOUCH_DELAY_S", "4")
        )
        self._chain_touch_stats = [0, 0]  # requests, keys
        if self._chain_touch:
            logger.info(
                "UMBP-CT chain-ordered touch on, final touch %.1f s after a request finishes",
                self._chain_touch_delay_s,
            )
        # Arrival prefetch (VLLM_UMBP_ARRIVAL_PREFETCH=1). A request that will load from the
        # pool has its objects copied SSD -> DRAM while it waits in the queue, oldest request
        # first, so the load at its first scheduling try reads DRAM. The copies of requests
        # that have not finished may hold at most VLLM_UMBP_PREFETCH_BUDGET_BYTES of DRAM
        # (counted as external tokens x VLLM_UMBP_PREFETCH_BYTES_PER_TOKEN); the server's own
        # budget (UMBP_DISTRIBUTED_PREFETCH_BUDGET_BYTES) must be larger, or it drops copies
        # that are still waiting to be read.
        self._pf_enabled = os.environ.get("VLLM_UMBP_ARRIVAL_PREFETCH", "0") == "1"
        self._pf_budget_tokens = int(
            float(os.environ.get("VLLM_UMBP_PREFETCH_BUDGET_BYTES", str(16 << 30)))
            / float(os.environ.get("VLLM_UMBP_PREFETCH_BYTES_PER_TOKEN", "19000"))
        )
        # Requests whose queue wait is shorter than a copy gain nothing from it.
        self._pf_min_tokens = int(os.environ.get("VLLM_UMBP_PREFETCH_MIN_TOKENS", "0"))
        # A copy made too early is gone before it is read: the DRAM entry tier turns over in
        # tens of seconds. So a copy starts only when at most this many requests are ahead
        # of its request in arrival order (requests run first come, first served); with a
        # short queue that is the moment the request arrives.
        self._pf_lead_reqs = int(os.environ.get("VLLM_UMBP_PREFETCH_LEAD_REQS", "12"))
        # Every request that has arrived and has not been admitted yet, oldest first.
        self._pf_waiting: dict[str, None] = {}
        self._pf_entries: dict[str, _ArrivalPrefetch] = {}
        self._pf_outstanding_tokens = 0
        self._pf_executor = (
            ThreadPoolExecutor(
                max_workers=int(os.environ.get("VLLM_UMBP_PREFETCH_WORKERS", "4")),
                thread_name_prefix="umbp-prefetch",
            )
            if self._pf_enabled
            else None
        )
        self._pf_stats: dict[str, float] = {}
        self._pf_log_t0 = time.monotonic()
        if self._pf_enabled:
            logger.info(
                "UMBP-PF arrival prefetch on: budget=%d tokens min=%d tokens lead=%d reqs",
                self._pf_budget_tokens, self._pf_min_tokens, self._pf_lead_reqs,
            )
        self._step_budget = vllm_config.scheduler_config.max_num_batched_tokens
        # block hash -> first request carrying it; request -> finish time
        self._hit_log_seen: dict[Any, str] = {}
        self._hit_log_finished: dict[str, float] = {}
        self._hit_log_local: dict[str, int] = {}
        self._hit_log_done: set[str] = set()
        partial_hash_hits = extra.get("enable_partial_hash_hits")
        if partial_hash_hits is None:
            # Mamba "align" checkpoints sit on prompt-tail hash boundaries, which
            # rarely fall on a cache block edge: with block-granular lookup a
            # hybrid model would almost never hit.
            partial_hash_hits = (
                bool(self._boundary_handoff_groups)
                and self.hash_block_size < self.block_size
            )
        self.enable_partial_hash_hits = bool(partial_hash_hits)
        lookup_groups = [
            kv_cache_config.kv_cache_groups[group_id]
            for group_id in kv_cache_config.prefix_cacheable_group_ids
        ]
        model_config = getattr(vllm_config, "model_config", None)
        speculative_config = getattr(vllm_config, "speculative_config", None)
        use_eagle = bool(
            speculative_config is not None and speculative_config.use_eagle_block_drop()
        )
        self.use_eagle = use_eagle
        explicit_eagle_groups = {
            group_id
            for group_id, group in enumerate(kv_cache_config.kv_cache_groups)
            if group.is_eagle_group
        }
        self._eagle_store_groups = (
            explicit_eagle_groups
            if explicit_eagle_groups or not use_eagle
            else set(kv_cache_config.prefix_cacheable_group_ids)
        )
        max_model_len = getattr(
            model_config,
            "max_model_len",
            kv_cache_config.num_blocks * self.block_size,
        )
        lookup_config = replace(
            kv_cache_config,
            kv_cache_groups=lookup_groups,
            num_blocks=1,
        )
        self._lookup_coordinator = (
            get_kv_cache_coordinator(
                kv_cache_config=lookup_config,
                max_model_len=max_model_len,
                max_in_flight_tokens=vllm_config.max_in_flight_tokens,
                use_eagle=use_eagle,
                enable_caching=True,
                enable_kv_cache_events=False,
                dcp_world_size=dcp_size,
                pcp_world_size=getattr(
                    vllm_config.parallel_config,
                    "prefill_context_parallel_size",
                    1,
                ),
                scheduler_block_size=self.block_size,
                hash_block_size=self.hash_block_size,
                # As core's scheduler builds its coordinator.
                num_prefill_lookahead=vllm_config.num_prefill_lookahead_tokens,
                allow_partial_hash_hits=self.enable_partial_hash_hits,
            )
            if lookup_groups
            and (
                len(lookup_groups) > 1
                or not isinstance(
                    _unwrap_store_spec(lookup_groups[0].kv_cache_spec),
                    FullAttentionSpec,
                )
            )
            else None
        )
        self._lookup_pool = _LookupBlockPool(self.hash_block_size)
        self._local_cache_lookup: Callable[[int, bytes], bool] | None = None
        if self._lookup_coordinator is not None:
            self._lookup_coordinator.block_pool = self._lookup_pool
            for manager in self._lookup_coordinator.single_type_managers:
                manager.block_pool = self._lookup_pool
                manager._null_block = self._lookup_pool.null_block
        self._pending_loads: dict[str, list[BlockTransferPlan] | BlockLoadBatch] = {}
        self._load_specs: dict[str, LoadSpec] = {}
        self._lookup_units = {
            group_id: (
                self.hash_block_size
                if self.enable_partial_hash_hits
                and (
                    self._lookup_coordinator is None
                    or self._lookup_coordinator.enable_partial_hash_hits
                )
                else self.group_block_sizes[group_id]
            )
            for group_id in kv_cache_config.prefix_cacheable_group_ids
        }
        # Groups whose hits the lookup coordinator backs off by one lookup unit.
        self._eagle_lookup_groups = (
            frozenset(
                group_id
                for group_id, manager in zip(
                    kv_cache_config.prefix_cacheable_group_ids,
                    self._lookup_coordinator.single_type_managers,
                )
                if manager.use_eagle
            )
            if self._lookup_coordinator is not None
            else frozenset()
        )
        self._pending_lookups: dict[str, _PendingLookup] = {}
        self._lookup_executor = ThreadPoolExecutor(
            max_workers=int(extra.get("lookup_workers", 2)),
            thread_name_prefix="umbp-lookup",
        )
        self._requests: dict[str, Request] = {}
        self._request_trackers: dict[str, RequestTracker] = {}
        self._pending_finished_stores: dict[str, list[BlockTransferPlan]] = {}
        self._gpu_block_pool: BlockPool | None = None
        self._store_events: dict[int, _StoreEvent] = {}
        self._kv_events: list[KVCacheEvent] = []
        self._event_objects: dict[tuple[int, bytes], _EventObject] = {}
        self._pending_event_objects: dict[tuple[int, bytes], int] = {}
        self._event_lookup: _EventLookup | None = None
        self._event_serial = 0
        self._next_event_reconcile = 0.0
        self._event_max_objects = extra.get("kv_event_max_objects", 65536)
        if type(self._event_max_objects) is not int or self._event_max_objects <= 0:
            raise ValueError("kv_event_max_objects must be a positive integer")
        self._event_check_batch_size = 256
        self._event_publications_suppressed = 0
        self._num_workers = getattr(vllm_config.parallel_config, "world_size", 1)
        self._store_event_counter = 0

    def bind_gpu_block_pool(self, gpu_block_pool: BlockPool) -> None:
        self._gpu_block_pool = gpu_block_pool

    def bind_local_cache_lookup(self, lookup: Callable[[int, bytes], bool]) -> None:
        self._local_cache_lookup = lookup

    def _has_local_block(self, group_id: int, block_hash: bytes) -> bool:
        if self._local_cache_lookup is not None:
            return self._local_cache_lookup(group_id, block_hash)
        pool = self._gpu_block_pool
        return pool is not None and bool(
            pool.get_cached_block(cast(BlockHash, block_hash), [group_id])
        )

    def bind_namespace(
        self, runtime: UMBPSchedulerHandle, codec: BlockIdentityCodec
    ) -> None:
        self.runtime = runtime
        self.codec = codec

    def get_num_new_matched_tokens(
        self, request: Request, num_computed_tokens: int
    ) -> tuple[int | None, bool]:
        log = self._step_log
        if log is None:
            return self._get_num_new_matched_tokens(request, num_computed_tokens)
        started = time.monotonic()
        result = self._get_num_new_matched_tokens(request, num_computed_tokens)
        log.add("gnnmt_s", time.monotonic() - started)
        log.add("gnnmt_n")
        if result[0] is None:
            log.add("gnnmt_none")
        return result

    def _get_num_new_matched_tokens(
        self, request: Request, num_computed_tokens: int
    ) -> tuple[int | None, bool]:
        if self._hit_log:
            self._hit_log_local[request.request_id] = num_computed_tokens
        if not self.enable_lookup:
            return 0, False
        if self.runtime is None:
            logger.warning_once(
                "UMBP lookup is disabled until workers report their namespace"
            )
            return 0, False
        hashes = tuple(request.block_hashes)
        pending = self._pending_lookups.get(request.request_id)
        if pending is not None:
            context = pending.context
            if (
                context.request is not request
                or context.local_tokens != num_computed_tokens
                or context.num_tokens != request.num_tokens
                or context.hashes != hashes
            ):
                self._pending_lookups.pop(request.request_id)
                pending.future.cancel()
                pending = None
        align = (
            self.hash_block_size if self.enable_partial_hash_hits else self.block_size
        )
        if not hashes or request.num_tokens < align:
            return 0, False
        if num_computed_tokens % self.block_size != 0:
            return 0, False
        group_ids = self.kv_cache_config.prefix_cacheable_group_ids
        if not group_ids:
            return 0, False
        # vLLM must execute at least the final prompt token. Returning a hit
        # for the entire prompt leaves the scheduler with no new token to run.
        max_external_token = request.num_tokens - 1
        if max_external_token <= num_computed_tokens:
            self._load_specs.pop(request.request_id, None)
            return 0, False
        if pending is not None and not pending.future.done():
            return None, False
        if pending is None:
            context = self._build_lookup_context(request, num_computed_tokens, hashes)
            if self.lookup_async:
                pending = _PendingLookup(
                    context,
                    self._lookup_executor.submit(self._lookup_all_ranks, context.keys),
                )
                if self._step_log is not None:
                    submitted, log = time.monotonic(), self._step_log
                    pending.future.add_done_callback(
                        lambda _f: log.lookup_done(time.monotonic() - submitted)
                    )
                if not self._await_lookup(pending.future):
                    self._pending_lookups[request.request_id] = pending
                    return None, False
        else:
            context = self._pending_lookups.pop(request.request_id).context
        try:
            hits = list(
                pending.future.result()
                if pending is not None
                else self._lookup_all_ranks(context.keys)
            )
        except Exception as exc:
            logger.debug("UMBP lookup failed request=%s: %s", request.request_id, exc)
            return 0, False
        keys = context.keys
        logical_objects_by_group = context.objects
        if len(hits) != len(keys):
            logger.debug(
                "UMBP lookup returned an invalid result length request=%s",
                request.request_id,
            )
            return 0, False
        per_rank = self.topology.rank_count
        offset = 0
        logical_hits_by_group: dict[int, list[bool]] = {}
        for group_id in group_ids:
            group_size = len(logical_objects_by_group[group_id]) * per_rank
            group_hits = hits[offset : offset + group_size]
            offset += group_size
            logical_hits_by_group[group_id] = (
                group_hits
                if per_rank == 1
                else [
                    all(group_hits[index : index + per_rank])
                    for index in range(0, len(group_hits), per_rank)
                ]
            )
        if self._lookup_coordinator is None:
            group_id = group_ids[0]
            group_hits = logical_hits_by_group[group_id]
            units_per_block = (
                self.group_block_sizes[group_id] // self._lookup_units[group_id]
            )
            matched_units = 0
            for index in range(units_per_block - 1, len(group_hits), units_per_block):
                if not group_hits[index]:
                    break
                matched_units = index + 1
            # Interior hashes are sparse: only the saved tail needs to exist.
            for index in range(
                min(matched_units + units_per_block - 1, len(group_hits)) - 1,
                matched_units - 1,
                -1,
            ):
                if group_hits[index]:
                    matched_units = index + 1
                    break
            if self.use_eagle:
                # As core's coordinator: the drafter recomputes the last block.
                matched_units = max(0, matched_units - units_per_block)
            need_to_load = matched_units * self._lookup_units[group_id]
            block_hashes_by_group: tuple[tuple[bytes | None, ...], ...] = ()
        else:
            need_to_load, block_hashes_by_group = self._coordinate_external_hits(
                list(context.hashes),
                num_computed_tokens,
                max_external_token - num_computed_tokens,
                logical_objects_by_group,
                logical_hits_by_group,
            )
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "UMBP lookup result request=%s keys=%s group_hits=%s load_tokens=%d",
                request.request_id,
                keys,
                logical_hits_by_group,
                need_to_load,
            )
        matched_tokens = num_computed_tokens + need_to_load
        if need_to_load <= 0:
            return 0, False
        self._load_specs[request.request_id] = LoadSpec(
            local_tokens=num_computed_tokens,
            external_tokens=matched_tokens,
            block_hashes_by_group=block_hashes_by_group,
        )
        load_async = self.load_async
        if block_hashes_by_group and not any(
            block_hash is not None
            for group_id, group_hashes in zip(
                group_ids, block_hashes_by_group, strict=True
            )
            for index, block_hash in enumerate(group_hashes)
            if index >= num_computed_tokens // self.group_block_sizes[group_id]
        ):
            load_async = False
        return need_to_load, load_async

    def _await_lookup(self, future: Future[Sequence[bool]]) -> bool:
        """Whether ``future`` finished within this pass's remaining wait budget."""
        timeout = min(self._lookup_wait_s, self._lookup_budget_left_s)
        if timeout <= 0:
            return future.done()
        started = time.monotonic()
        wait_futures((future,), timeout=timeout)
        waited = time.monotonic() - started
        self._lookup_budget_left_s -= waited
        if self._step_log is not None:
            self._step_log.add("await_s", waited)
            self._step_log.add("await_n")
        return future.done()

    def _lookup_all_ranks(
        self,
        keys: Sequence[str],
        lookup: Callable[[Sequence[str]], Sequence[bool]] | None = None,
    ) -> list[bool]:
        """Existence of every rank's copy of each object in ``keys``.

        ``keys`` holds ``rank_count`` consecutive keys per object, first rank
        first. Ranks store an object together, and a hash-granular lookup probes
        far more positions than hold anything, so only the first rank is asked
        about every object and the others only about the objects it holds.
        Whether every rank holds an object, the only answer callers use, equals
        one query for all keys.
        """
        per_rank = self.topology.rank_count
        if lookup is None:
            lookup = self.runtime.lookup
        if per_rank == 1 or len(keys) % per_rank:
            return list(lookup(keys))
        probe = keys[::per_rank]
        first = list(lookup(probe))
        if len(first) != len(probe):
            raise RuntimeError(f"{len(first)} lookup results for {len(probe)} keys")
        held = [index for index, exists in enumerate(first) if exists]
        hits = [False] * len(keys)
        if not held:
            return hits
        others = [
            keys[index * per_rank + rank]
            for index in held
            for rank in range(1, per_rank)
        ]
        rest = list(lookup(others))
        if len(rest) != len(others):
            raise RuntimeError(f"{len(rest)} lookup results for {len(others)} keys")
        width = per_rank - 1
        for position, index in enumerate(held):
            base = index * per_rank
            hits[base] = True
            hits[base + 1 : base + per_rank] = rest[
                position * width : (position + 1) * width
            ]
        return hits

    # ---- arrival prefetch -------------------------------------------------------------

    def _pf_count(self, key: str, value: float = 1.0) -> None:
        self._pf_stats[key] = self._pf_stats.get(key, 0.0) + value

    def _pf_disable(self, where: str, exc: BaseException) -> None:
        """A bug in this experiment must not take the engine down: log once and stop."""
        self._pf_enabled = False
        logger.exception("UMBP-PF disabled after an error in %s: %r", where, exc)

    def on_new_request(self, request: Request) -> None:
        """Core thread, when a request arrives: start the lookup of its pool objects."""
        if not self._pf_enabled:
            return
        try:
            self._pf_on_new_request(request)
        except Exception as exc:  # noqa: BLE001
            self._pf_disable("on_new_request", exc)

    def _pf_on_new_request(self, request: Request) -> None:
        if self.runtime is None or self.codec is None:
            return
        if not self.enable_lookup or getattr(self.runtime, "prefetch", None) is None:
            return
        self._pf_waiting[request.request_id] = None
        hashes = tuple(request.block_hashes)
        align = (
            self.hash_block_size if self.enable_partial_hash_hits else self.block_size
        )
        if not hashes or request.num_tokens < max(align, self._pf_min_tokens):
            return
        local = self._pf_local_estimate(hashes, request.num_tokens)
        if request.num_tokens - 1 - local < max(align, self._pf_min_tokens):
            self._pf_count("skip_local")
            return
        entry = _ArrivalPrefetch(request, hashes, time.monotonic(), local)
        assert self._pf_executor is not None
        try:
            entry.lookup = self._pf_executor.submit(self._pf_lookup, entry)
        except RuntimeError:  # executor shut down
            return
        self._pf_entries[request.request_id] = entry
        self._pf_count("arrivals")

    def _pf_lookup(self, entry: _ArrivalPrefetch) -> list[bool]:
        """Prefetch thread: which pool objects past the GPU-held prefix exist."""
        # Built here, not on the core thread: a long prompt has ~10^5 keys.
        entry.context = self._build_lookup_context(
            entry.request, entry.local_tokens, entry.hashes
        )
        lookup = getattr(self.runtime, "lookup_direct", None) or self.runtime.lookup
        return self._lookup_all_ranks(entry.context.keys, lookup)

    def _pf_local_estimate(self, hashes: tuple[bytes, ...], num_tokens: int) -> int:
        """Leading tokens the GPU cache holds now, in whole blocks (attention groups)."""
        group_ids = self.kv_cache_config.prefix_cacheable_group_ids
        groups = [g for g in group_ids if g not in self._boundary_handoff_groups]
        if not groups:
            return 0
        tokens = 0
        step = max(self.group_block_sizes[g] for g in groups)
        for end in range(step, num_tokens, step):
            index = end // self.hash_block_size - 1
            if index >= len(hashes) or not all(
                self._has_local_block(g, hashes[index]) for g in groups
            ):
                break
            tokens = end
        return tokens - tokens % self.block_size

    def _pf_drop(self, request_id: str) -> None:
        self._pf_waiting.pop(request_id, None)
        entry = self._pf_entries.pop(request_id, None)
        if entry is None:
            return
        entry.dead = True
        if entry.lookup is not None:
            entry.lookup.cancel()
        self._pf_outstanding_tokens -= entry.tokens
        entry.tokens = 0

    def _pf_admitted(self, request_id: str) -> None:
        """The request reached the scheduler: a copy that has not started is too late."""
        self._pf_waiting.pop(request_id, None)
        entry = self._pf_entries.get(request_id)
        if entry is None:
            return
        if entry.copy is None:
            self._pf_count("late")
            self._pf_drop(request_id)

    def _pf_pump(self) -> None:
        """Core thread, once per step: plan finished lookups, start copies that are due."""
        try:
            self._pf_pump_once()
        except Exception as exc:  # noqa: BLE001
            self._pf_disable("pump", exc)

    def _pf_pump_once(self) -> None:
        if not self._pf_entries:
            self._pf_log()
            return
        planned = 0
        for ahead, request_id in enumerate(self._pf_waiting):  # arrival order
            if ahead >= self._pf_lead_reqs:
                break
            entry = self._pf_entries.get(request_id)
            if entry is None or entry.copy is not None:
                continue
            if entry.lookup is None or not entry.lookup.done():
                continue
            if entry.tokens == 0:
                if planned >= 4:
                    break
                planned += 1
                keys, tokens = self._pf_plan(entry)
                if not keys:
                    self._pf_count("skip_nohit")
                    self._pf_entries.pop(request_id, None)
                    entry.dead = True
                    continue
                entry.tokens, entry.keys = tokens, len(keys)
                entry.context = keys  # the lookup context is no longer needed
                self._pf_outstanding_tokens += tokens
            if (
                self._pf_outstanding_tokens > self._pf_budget_tokens
                and self._pf_outstanding_tokens != entry.tokens
            ):
                # Wait for credit; it returns when older requests finish.
                self._pf_count("budget_waits")
                break
            entry.started = time.monotonic()
            assert self._pf_executor is not None
            try:
                entry.copy = self._pf_executor.submit(self._pf_copy, request_id, entry)
                self._pf_count("copies")
            except RuntimeError:
                self._pf_drop(request_id)
        self._pf_log()

    def _pf_plan(self, entry: _ArrivalPrefetch) -> tuple[list[str], int]:
        """Every rank's keys of the objects the request's pool hit would load now."""
        context = entry.context
        try:
            hits = list(entry.lookup.result()) if entry.lookup is not None else []
        except Exception:  # noqa: BLE001
            return [], 0
        if len(hits) != len(context.keys):
            return [], 0
        group_ids = self.kv_cache_config.prefix_cacheable_group_ids
        per_rank = self.topology.rank_count
        offset = 0
        logical_hits: dict[int, list[bool]] = {}
        for group_id in group_ids:
            size = len(context.objects[group_id]) * per_rank
            group_hits = hits[offset : offset + size]
            offset += size
            logical_hits[group_id] = (
                group_hits
                if per_rank == 1
                else [
                    all(group_hits[i : i + per_rank])
                    for i in range(0, len(group_hits), per_rank)
                ]
            )
        if self._lookup_coordinator is None:
            return [], 0
        local = entry.local_tokens
        try:
            need, block_hashes_by_group = self._coordinate_external_hits(
                list(entry.hashes),
                local,
                entry.request.num_tokens - 1 - local,
                context.objects,
                logical_hits,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "UMBP-PF plan failed req=%s: %s", entry.request.request_id, exc
            )
            return [], 0
        if need <= 0 or not block_hashes_by_group:
            return [], 0
        assert self.codec is not None
        keys: list[str] = []
        for group_index, group_id in enumerate(group_ids):
            start_block = local // self.group_block_sizes[group_id]
            for block_index, block_hash in enumerate(
                block_hashes_by_group[group_index]
            ):
                if block_index >= start_block and block_hash is not None:
                    keys.extend(
                        _rank_keys(self.codec, self.topology, block_hash, group_id)
                    )
        return list(dict.fromkeys(keys)), need

    def _pf_copy(self, request_id: str, entry: _ArrivalPrefetch) -> None:
        """Prefetch thread: the copy itself (blocks for the SSD reads)."""
        keys = entry.context
        try:
            copied = self.runtime.prefetch(keys) if not entry.dead else 0
        except Exception as exc:  # noqa: BLE001
            logger.warning("UMBP-PF copy failed req=%s: %s", request_id, exc)
            copied = -1
        now = time.monotonic()
        logger.info(
            "UMBP-PF copy req=%s tokens=%d local=%d keys=%d copied=%d rpc_s=%.3f "
            "plan_wait_s=%.3f dead=%d",
            request_id, entry.tokens, entry.local_tokens, entry.keys, copied,
            now - entry.started, entry.started - entry.arrived, int(entry.dead),
        )

    def _pf_log(self) -> None:
        now = time.monotonic()
        if now - self._pf_log_t0 < 10.0:
            return
        stats = self._pf_stats
        copying = sum(
            1
            for e in self._pf_entries.values()
            if e.copy is not None and not e.copy.done()
        )
        logger.info(
            "UMBP-PF dt=%.1f arrivals=%d skip_local=%d skip_nohit=%d copies=%d late=%d "
            "used=%d budget_waits=%d waiting=%d entries=%d copying=%d outstanding_tokens=%d",
            now - self._pf_log_t0, stats.get("arrivals", 0), stats.get("skip_local", 0),
            stats.get("skip_nohit", 0), stats.get("copies", 0), stats.get("late", 0),
            stats.get("used", 0), stats.get("budget_waits", 0), len(self._pf_waiting),
            len(self._pf_entries), copying, self._pf_outstanding_tokens,
        )
        self._pf_stats = {}
        self._pf_log_t0 = now

    def _local_hit_keys(self, request: Any, hit_tokens: int) -> list[str]:
        """Every rank's objects that back ``hit_tokens`` of local prefix hit.

        A prefix the GPU cache serves is never read back from the pool, so its
        copies there would age from their first store and be evicted first,
        then lost in both tiers once the GPU evicts it too.
        """
        hashes = request.block_hashes
        unit = self.hash_block_size
        if hit_tokens <= 0 or not hashes:
            return []
        keys: list[str] = []
        by_end: list[tuple[int, tuple[str, ...]]] = []
        for group_id in self.kv_cache_config.prefix_cacheable_group_ids:
            if group_id in self._boundary_handoff_groups:
                # The hit resumes from the state checkpoint at its end. The retained
                # states below it are where later requests branching inside this
                # prefix resume; a GPU hit never reads them back from the pool, so
                # without a touch they age from their first store and go first.
                step = self._retention_interval
                if step is None:
                    step = self.block_size
                ends: Sequence[int] = (
                    list(range(step, hit_tokens, step)) if step else []
                ) + [hit_tokens]
            else:
                block = self.group_block_sizes[group_id]
                ends = list(range(block, hit_tokens + 1, block))
                if hit_tokens % block and self.enable_partial_hash_hits:
                    # The tail page is keyed at the prompt-tail boundary it was
                    # stored at: the hit's end, or one or two units past it
                    # when the drafter recomputes the last unit.
                    ends.extend(range(hit_tokens, hit_tokens + 2 * unit + 1, unit))
            for end in ends:
                index = end // unit - 1
                if 0 <= index < len(hashes):
                    by_end.append(
                        (end, _rank_keys(self.codec, self.topology, hashes[index], group_id))
                    )
        if self._chain_touch:
            # Deepest first: the pool counts later keys of one hint as more recent.
            by_end.sort(key=lambda item: -item[0])
        for _, rank_keys in by_end:
            keys.extend(rank_keys)
        return list(dict.fromkeys(keys))

    def _chain_touch_finished(self, request: Request) -> None:
        """Touch a finished request's whole prompt, deepest page first, once its stores
        have had time to land."""
        touch = getattr(self.runtime, "touch", None)
        if not callable(touch) or self.codec is None:
            return
        unit = self.hash_block_size
        keys = self._local_hit_keys(request, request.num_prompt_tokens // unit * unit)
        if not keys:
            return
        self._chain_touch_stats[0] += 1
        self._chain_touch_stats[1] += len(keys)

        def run() -> None:
            try:
                touch(keys)
            except Exception as exc:  # noqa: BLE001
                logger.debug("UMBP-CT touch failed: %s", exc)

        timer = threading.Timer(self._chain_touch_delay_s, run)
        timer.daemon = True
        timer.start()
        if self._chain_touch_stats[0] % 200 == 0:
            logger.info(
                "UMBP-CT final touches: %d requests, %d keys",
                self._chain_touch_stats[0], self._chain_touch_stats[1],
            )

    def _touch_async(self, keys: list[str]) -> None:
        touch = getattr(self.runtime, "touch", None)
        if not keys or not callable(touch):
            return
        # The executor is shut down at close; a lost recency hint is harmless.
        with contextlib.suppress(RuntimeError):
            self._lookup_executor.submit(touch, keys)

    def _build_lookup_context(
        self, request: Request, local_tokens: int, hashes: tuple[bytes, ...]
    ) -> _LookupContext:
        assert self.codec is not None
        objects: dict[int, list[tuple[bytes, tuple[str, ...]]]] = {}
        keys: list[str] = []
        lookup_limit = request.num_tokens - 1
        if self.use_eagle and self._lookup_coordinator is not None:
            lookup_limit = min(request.num_tokens, len(hashes) * self.hash_block_size)
        for group_id, unit in self._lookup_units.items():
            group_objects = []
            for token_end in range(local_tokens + unit, lookup_limit + 1, unit):
                block_hash = self._object_hash_at_token_end(hashes, token_end)
                rank_keys = _rank_keys(self.codec, self.topology, block_hash, group_id)
                group_objects.append((block_hash, rank_keys))
                keys.extend(rank_keys)
            objects[group_id] = group_objects
        return _LookupContext(
            request, local_tokens, request.num_tokens, hashes, objects, keys
        )

    def _coordinate_external_hits(
        self,
        request_hashes: list[bytes],
        local_tokens: int,
        max_external_tokens: int,
        logical_objects_by_group: dict[int, list[tuple[bytes, tuple[str, ...]]]],
        logical_hits_by_group: dict[int, list[bool]],
    ) -> tuple[int, tuple[tuple[bytes | None, ...], ...]]:
        """Apply the core KV cache coordinator to runtime lookup results."""
        coordinator = self._lookup_coordinator
        assert coordinator is not None
        pool = self._lookup_pool
        try:
            pool.hits = [
                {
                    block_hash
                    for (block_hash, _), hit in zip(
                        logical_objects_by_group[group_id],
                        logical_hits_by_group[group_id],
                        strict=True,
                    )
                    if hit
                }
                for group_id in self.kv_cache_config.prefix_cacheable_group_ids
            ]
            if local_tokens:
                for group_index, group_id in enumerate(
                    self.kv_cache_config.prefix_cacheable_group_ids
                ):
                    for token_end in range(
                        self._lookup_units[group_id],
                        local_tokens + 1,
                        self._lookup_units[group_id],
                    ):
                        block_hash = self._object_hash_at_token_end(
                            request_hashes, token_end
                        )
                        if self._has_local_block(group_id, block_hash):
                            pool.hits[group_index].add(block_hash)
            hit_blocks, hit_length, _ = coordinator.find_longest_cache_hit(
                cast(list[BlockHash], request_hashes),
                local_tokens + max_external_tokens,
            )
            block_hashes_by_group = tuple(
                tuple(
                    None
                    if block.is_null or block.block_hash is None
                    else get_block_hash(block.block_hash)
                    for block in group_blocks
                )
                for group_blocks in hit_blocks
            )
            return max(0, hit_length - local_tokens), block_hashes_by_group
        finally:
            self._reset_lookup_coordinator()

    def _reset_lookup_coordinator(self) -> None:
        self._lookup_pool.clear_lookup()

    def update_state_after_alloc(
        self,
        request: Request,
        blocks: KVCacheBlocks,
        num_external_tokens: int,
    ) -> None:
        if self._pf_enabled:
            try:
                entry = self._pf_entries.get(request.request_id)
                if entry is not None and entry.copy is not None and num_external_tokens > 0:
                    self._pf_count("used")
                    logger.info(
                        "UMBP-PF use req=%s planned=%d loaded=%d copy_done=%d age_s=%.1f",
                        request.request_id, entry.tokens, num_external_tokens,
                        int(entry.copy.done()), time.monotonic() - entry.arrived,
                    )
                elif num_external_tokens > 0:
                    logger.info(
                        "UMBP-PF miss req=%s loaded=%d had_entry=%d",
                        request.request_id, num_external_tokens, int(entry is not None),
                    )
                self._pf_admitted(request.request_id)
            except Exception as exc:  # noqa: BLE001
                self._pf_disable("admission", exc)
        if self._hit_log:
            self._log_hit(request, num_external_tokens)
        self._requests[request.request_id] = request
        block_groups = blocks.get_block_ids(
            group_ids=self.kv_cache_config.prefix_cacheable_group_ids
        )
        if not block_groups:
            return
        tracker = self._tracker_for_request(request.request_id)
        # Core passes the request's whole table on every admission. A request
        # whose load failed is recomputed into newly allocated blocks.
        tracker.replace_blocks(tuple(list(group) for group in block_groups))
        if num_external_tokens <= 0:
            self._pending_loads.pop(request.request_id, None)
            self._load_specs.pop(request.request_id, None)
            return
        spec = self._load_specs.get(request.request_id)
        if spec is not None:
            tracker.load_spec = spec
            # A request admitted without its load would wait for it forever.
            assert num_external_tokens == spec.num_tokens_to_load, (
                f"UMBP matched {spec.num_tokens_to_load} external tokens for "
                f"request {request.request_id} but {num_external_tokens} were "
                "allocated"
            )
        plans = self._load_plans_for_external_tokens(
            request,
            tracker,
            block_groups,
            num_external_tokens,
        )
        self._pending_loads[request.request_id] = plans

    def build_connector_meta(
        self, scheduler_output: SchedulerOutput
    ) -> KVConnectorMetadata:
        log = self._step_log
        if log is None:
            return self._build_connector_meta(scheduler_output)
        started = time.monotonic()
        meta = self._build_connector_meta(scheduler_output)
        log.add("bcm_s", time.monotonic() - started)
        scheduled = scheduler_output.num_scheduled_tokens
        log.step(
            sum(scheduled.values()),
            len(scheduled),
            len(getattr(meta, "load_requests", ()) or ()),
            self._step_budget,
        )
        return meta

    def _build_connector_meta(
        self, scheduler_output: SchedulerOutput
    ) -> KVConnectorMetadata:
        meta = UMBPConnectorMetadata(async_load=self.load_async)
        # Lookups of this schedule() pass are done; the next pass gets a new budget.
        self._lookup_budget_left_s = self._lookup_pass_budget_s
        reserved_store_keys = {
            key for event in self._store_events.values() for key in event.iter_keys()
        }
        if self._pf_enabled:
            for request_id in scheduler_output.finished_req_ids:
                self._pf_drop(request_id)
            for new_request in scheduler_output.scheduled_new_reqs:
                self._pf_waiting.pop(new_request.req_id, None)
            self._pf_pump()
        for request_id in scheduler_output.finished_req_ids:
            self._pending_loads.pop(request_id, None)
            self._load_specs.pop(request_id, None)
            self._request_trackers.pop(request_id, None)
            self._requests.pop(request_id, None)
            pending = self._pending_lookups.pop(request_id, None)
            if pending is not None:
                pending.future.cancel()
        touch_keys: list[str] = []
        for request in scheduler_output.scheduled_new_reqs:
            load_plans = self._pending_loads.pop(request.req_id, [])
            self._load_specs.pop(request.req_id, None)
            tracked_request = self._requests.get(request.req_id)
            if self.enable_lookup and tracked_request is not None:
                # Loaded objects are refreshed by the read itself; only the part
                # the GPU cache served needs a recency hint.
                tracker = self._request_trackers.get(request.req_id)
                spec = tracker.load_spec if tracker is not None else None
                local_hit = (
                    spec.local_tokens
                    if load_plans and spec is not None
                    else request.num_computed_tokens
                )
                touch_keys.extend(self._local_hit_keys(tracked_request, local_hit))
            if load_plans:
                meta.load_requests[request.req_id] = load_plans
            else:
                tracker = self._tracker_for_request(request.req_id)
                total_tokens = (
                    request.num_computed_tokens
                    + scheduler_output.num_scheduled_tokens[request.req_id]
                )
                selected_block_ids = tuple(
                    request.block_ids[group_id]
                    for group_id in (self.kv_cache_config.prefix_cacheable_group_ids)
                )
                store_plans = (
                    None
                    if not self.store_enabled or tracked_request is None
                    else self._store_plans(
                        tracked_request,
                        tracker,
                        total_tokens,
                        block_ids_override=selected_block_ids,
                        skip_keys=reserved_store_keys,
                    )
                )
                logger.debug(
                    "UMBP store planning request=%s tokens=%d plans=%d keys=%s",
                    request.req_id,
                    total_tokens,
                    len(store_plans) if store_plans is not None else 0,
                    tuple(store_plans.keys) if store_plans is not None else (),
                )
                if store_plans:
                    meta.store_requests[request.req_id] = store_plans
                    reserved_store_keys.update(store_plans.keys)
        self._touch_async(touch_keys)

        block_state = getattr(scheduler_output, "kv_connector_block_state", None)
        cached_reqs = scheduler_output.scheduled_cached_reqs
        for request_id in cached_reqs.req_ids:
            load_plans = self._pending_loads.pop(request_id, [])
            self._load_specs.pop(request_id, None)
            if load_plans:
                meta.load_requests[request_id] = load_plans
            elif self.store_enabled:
                cached_request = self._requests.get(request_id)
                cached_tracker = self._request_trackers.get(request_id)
                if cached_request is not None and cached_tracker is not None:
                    request_index = cached_reqs.req_ids.index(request_id)
                    num_computed_tokens = cached_reqs.num_computed_tokens[request_index]
                    # num_tokens counts sampled tokens, so a decode step is still
                    # short of it; only the prompt bounds the prefill.
                    is_prefill = num_computed_tokens < cached_request.num_prompt_tokens
                    if not is_prefill and not self.save_decode_cache:
                        continue
                    new_block_ids = cached_reqs.new_block_ids[request_index]
                    if new_block_ids:
                        selected = tuple(
                            new_block_ids[group_id]
                            for group_id in (
                                self.kv_cache_config.prefix_cacheable_group_ids
                            )
                        )
                        cached_tracker.update_blocks(selected)
                    # The appended table keeps blocks core has since freed, such
                    # as sliding-window blocks out of the window, which a store
                    # would read; core's current table does not.
                    current = (
                        block_state.get_block_ids(request_id)
                        if block_state is not None
                        else None
                    )
                    if current is not None:
                        cached_tracker.block_ids = tuple(
                            list(current[group_id])
                            for group_id in (
                                self.kv_cache_config.prefix_cacheable_group_ids
                            )
                        )
                    total_tokens = (
                        num_computed_tokens
                        + scheduler_output.num_scheduled_tokens[request_id]
                    )
                    if not self.save_decode_cache:
                        total_tokens = min(
                            total_tokens, cached_request.num_prompt_tokens
                        )
                    store_plans = self._store_plans(
                        cached_request,
                        cached_tracker,
                        total_tokens,
                        block_ids_override=cached_tracker.block_ids,
                        skip_keys=reserved_store_keys,
                    )
                    if store_plans:
                        meta.store_requests[request_id] = store_plans
                        reserved_store_keys.update(store_plans.keys)

        # Async loads are admitted without scheduling model tokens, so those
        # requests are absent from both scheduled request collections above.
        # Forward their plans explicitly so workers can start the transfer and
        # eventually move the request out of WAITING_FOR_REMOTE_KVS.
        for request_id, load_plans in list(self._pending_loads.items()):
            self._pending_loads.pop(request_id, None)
            self._load_specs.pop(request_id, None)
            if load_plans:
                meta.load_requests[request_id] = load_plans

        for request_id in scheduler_output.preempted_req_ids or set():
            meta.preempted_request_ids.add(request_id)
            self._load_specs.pop(request_id, None)
            if preempted_tracker := self._request_trackers.get(request_id):
                preempted_tracker.reset()

        if (
            self.store_enabled
            and block_state is not None
            and block_state.boundary_state_offloads
        ):
            self._add_boundary_state_plans(
                block_state.boundary_state_offloads,
                meta,
                set(scheduler_output.finished_req_ids),
                set(scheduler_output.preempted_req_ids or ()),
            )

        finished_stores = self._pending_finished_stores
        self._pending_finished_stores = {}
        for plans in finished_stores.values():
            meta.store_plans.extend(plans)
        event_plan_groups = tuple(meta.store_plan_groups())
        if event_plan_groups:
            event = self._store_event_counter
            self._store_event_counter += 1
            meta.store_event = event
            self._store_events[event] = _StoreEvent(event_plan_groups)
            if self._gpu_block_pool is not None:
                self._gpu_block_pool.touch(
                    [
                        self._gpu_block_pool.blocks[block_id]
                        for block_id in dict.fromkeys(
                            block_id
                            for plans in event_plan_groups
                            for block_id in (
                                plans.block_ids
                                if isinstance(plans, BlockStoreBatch)
                                else (plan.block_id for plan in plans)
                            )
                        )
                    ]
                )
                # Transfer ownership from request-finish pins to the event.
                for plans in finished_stores.values():
                    self._gpu_block_pool.free_blocks(
                        self._gpu_block_pool.blocks[block_id]
                        for block_id in dict.fromkeys(plan.block_id for plan in plans)
                    )
        return meta

    def _add_boundary_state_plans(
        self,
        offloads: dict[str, list[tuple[int, int, int]]],
        meta: UMBPConnectorMetadata,
        finished: set[str],
        preempted: set[str],
    ) -> None:
        """Store exact Mamba/hybrid boundary blocks handed off by core."""
        if self.codec is None:
            return
        prefix_groups = set(self.kv_cache_config.prefix_cacheable_group_ids)
        for request_id, entries in offloads.items():
            if request_id in finished or request_id in preempted:
                logger.debug(
                    "UMBP boundary offloads dropped request=%s entries=%d: %s",
                    request_id,
                    len(entries),
                    "finished" if request_id in finished else "preempted",
                )
                continue
            request = self._requests.get(request_id)
            tracker = self._request_trackers.get(request_id)
            if request is None or tracker is None:
                logger.debug(
                    "UMBP boundary offloads dropped request=%s entries=%d: untracked",
                    request_id,
                    len(entries),
                )
                continue
            plans: list[BlockTransferPlan] = []
            window = self._window_boundaries(request)
            for group_id, block_id, boundary_tokens in entries:
                if group_id not in prefix_groups or block_id == NULL_BLOCK_ID:
                    continue
                if boundary_tokens <= 0:
                    continue
                hash_index = boundary_tokens // self.hash_block_size - 1
                if not 0 <= hash_index < len(request.block_hashes):
                    continue
                block_hash = request.block_hashes[hash_index]
                plans.append(
                    BlockTransferPlan(
                        key=self.codec.key(block_hash, group_id),
                        block_id=block_id,
                        request_id=request_id,
                        group_id=group_id,
                        block_hash=block_hash,
                        parent_block_hash=(
                            request.block_hashes[hash_index - 1]
                            if self.enable_kv_cache_events and hash_index > 0
                            else None
                        ),
                        block_size=self.group_block_sizes[group_id],
                        low_priority=boundary_tokens in window,
                    )
                )
            boundaries = {
                plan_boundary for _, _, plan_boundary in entries if plan_boundary > 0
            }
            for boundary_tokens in sorted(boundaries):
                tail = self._attention_tail_plans(
                    request_id, request, tracker.block_ids, boundary_tokens
                )
                if boundary_tokens in window:
                    tail = [replace(plan, low_priority=True) for plan in tail]
                plans.extend(tail)
            logger.debug(
                "UMBP boundary offloads request=%s offered=%d stored=%d boundaries=%s",
                request_id,
                len(entries),
                len(plans),
                sorted(boundaries),
            )
            if plans:
                meta.store_plans.extend(plans)

    def _window_boundaries(self, request: Any) -> frozenset[int]:
        """Tail-window state positions of `request`'s prompt (low class)."""
        if not self._window_block_size:
            return frozenset()
        window = mamba_tail_window()
        assert window is not None
        stride, width = window
        prompt = request.num_prompt_tokens
        # In-step states sit at most at `prompt - B`; a step end inside the
        # last block and the prompt-tail checkpoint stay class 0.
        first = max(prompt - width, 1)
        return frozenset(
            p
            for p in range(-(-first // stride) * stride, prompt - self._window_block_size + 1, stride)
            if p % self._window_block_size == 0 and p % self._full_block_size
        )

    def _attention_tail_plans(
        self,
        request_id: str,
        request: Any,
        block_ids: tuple[list[int], ...],
        boundary: int,
    ) -> list[BlockTransferPlan]:
        """Full-attention pages a hybrid prefix needs to resume at ``boundary``.

        Core hands off only the state pages of a boundary. When the boundary falls
        inside a full-attention block, that block's page holds the tokens before it,
        and without it the prefix cannot be restored there.
        """
        plans: list[BlockTransferPlan] = []
        for index, group_id in enumerate(
            self.kv_cache_config.prefix_cacheable_group_ids
        ):
            if group_id not in self._attention_tail_groups or index >= len(block_ids):
                continue
            group_block_size = self.group_block_sizes[group_id]
            end = self._attention_tail_end(group_id, boundary)
            hash_index = end // self.hash_block_size - 1
            # A block-aligned end is a full block, stored with the block chain.
            if end % group_block_size == 0 or not 0 <= hash_index < len(
                request.block_hashes
            ):
                continue
            block_index = end // group_block_size
            group_blocks = block_ids[index]
            if (
                block_index >= len(group_blocks)
                or group_blocks[block_index] == NULL_BLOCK_ID
            ):
                continue
            block_hash = request.block_hashes[hash_index]
            plans.append(
                BlockTransferPlan(
                    key=self.codec.key(block_hash, group_id),
                    block_id=group_blocks[block_index],
                    request_id=request_id,
                    group_id=group_id,
                    block_hash=block_hash,
                    parent_block_hash=(
                        request.block_hashes[hash_index - 1]
                        if self.enable_kv_cache_events and hash_index > 0
                        else None
                    ),
                    block_size=group_block_size,
                )
            )
        return plans

    def _attention_tail_end(self, group_id: int, boundary: int) -> int:
        """Where the attention page restoring ``boundary`` must be keyed.

        Eagle groups match one lookup unit past a candidate and then drop it, so
        their page has to answer the position one unit after the state page.
        """
        if group_id in self._eagle_lookup_groups:
            return boundary + self._lookup_units[group_id]
        return boundary

    def _tracker_for_request(self, request_id: str) -> RequestTracker:
        return self._request_trackers.setdefault(request_id, RequestTracker())

    def _load_plans_for_external_tokens(
        self,
        request: Any,
        tracker: RequestTracker,
        block_groups: tuple[list[int], ...],
        num_external_tokens: int,
    ) -> BlockLoadBatch:
        """Batch the logical objects selected by the external hit window."""
        assert self.codec is not None
        group_ids = self.kv_cache_config.prefix_cacheable_group_ids
        spec = tracker.load_spec
        local_tokens = spec.local_tokens if spec is not None else 0
        end_tokens = local_tokens + num_external_tokens
        keys: list[str] = []
        destinations: list[int] = []
        groups: list[int] = []
        if spec is not None and spec.block_hashes_by_group:
            for group_index, group_id in enumerate(group_ids):
                group_block_size = self.group_block_sizes[group_id]
                start_block = local_tokens // group_block_size
                for block_index, block_hash in enumerate(
                    spec.block_hashes_by_group[group_index]
                ):
                    if block_index < start_block or block_hash is None:
                        continue
                    block_id = block_groups[group_index][block_index]
                    if block_id == NULL_BLOCK_ID:
                        raise RuntimeError(
                            "UMBP external hit mapped a runtime object to a null "
                            f"destination block for group {group_id}"
                        )
                    keys.append(self.codec.key(block_hash, group_id=group_id))
                    destinations.append(block_id)
                    groups.append(group_id)
            return BlockLoadBatch(keys, destinations, groups, request.request_id)
        for group_index, group_id in enumerate(group_ids):
            group_block_size = self.group_block_sizes[group_id]
            start_block = local_tokens // group_block_size
            num_full_blocks = end_tokens // group_block_size
            for block_index in range(start_block, num_full_blocks):
                token_end = (block_index + 1) * group_block_size
                keys.append(
                    self.codec.key(
                        self._object_hash_at_token_end(request.block_hashes, token_end),
                        group_id=group_id,
                    )
                )
                destinations.append(block_groups[group_index][block_index])
                groups.append(group_id)
        partial_tokens = end_tokens % self.block_size
        if partial_tokens and self.enable_partial_hash_hits:
            hash_index = end_tokens // self.hash_block_size - 1
            for group_index, group_id in enumerate(group_ids):
                block_index = end_tokens // self.group_block_sizes[group_id]
                group_block_ids = block_groups[group_index]
                if (
                    block_index >= len(group_block_ids)
                    or group_block_ids[block_index] == NULL_BLOCK_ID
                ):
                    raise RuntimeError("UMBP partial hit has no destination block")
                keys.append(self.codec.key(request.block_hashes[hash_index], group_id))
                destinations.append(group_block_ids[block_index])
                groups.append(group_id)
        return BlockLoadBatch(keys, destinations, groups, request.request_id)

    def _object_hash_at_token_end(
        self, hashes: Sequence[bytes], token_end: int
    ) -> bytes:
        hash_index = token_end // self.hash_block_size - 1
        if hash_index < 0 or hash_index >= len(hashes):
            raise ValueError(
                f"request has {len(hashes)} hashes, needs index {hash_index}"
            )
        return hashes[hash_index]

    def _store_plans(
        self,
        request: Any,
        tracker: RequestTracker,
        token_count: int,
        block_ids_override: tuple[list[int], ...] | None = None,
        skip_keys: set[str] | None = None,
    ) -> BlockStoreBatch:
        """Describe full blocks produced by a scheduled prefill."""
        request_id = getattr(request, "request_id", None) or request.req_id
        if self.codec is None:
            return BlockStoreBatch([], [], [], [], [], [], [], request_id)
        group_ids = self.kv_cache_config.prefix_cacheable_group_ids
        # Scheduled speculative tokens are not yet verified and have no block
        # hash; a block is storable only once its hash exists.
        token_count = min(token_count, len(request.block_hashes) * self.hash_block_size)
        previous_saved_tokens = tracker.saved_tokens
        # Hash-aligned rather than block-aligned: a group with smaller blocks
        # than the largest must not wait for that group's block to fill.
        save_to = tracker.mark_saved(token_count, self.hash_block_size)
        keys: list[str] = []
        source_block_ids: list[int] = []
        plan_group_ids: list[int] = []
        block_hashes: list[bytes] = []
        parent_block_hashes: list[bytes | None] = []
        token_ids_by_block: list[tuple[int, ...]] = []
        block_sizes: list[int] = []
        block_groups = (
            block_ids_override
            if block_ids_override is not None
            else tuple(request.block_ids[group_id] for group_id in group_ids)
        )
        candidate_ranges = {
            group_id: range(
                previous_saved_tokens // self.group_block_sizes[group_id],
                min(save_to // self.group_block_sizes[group_id], len(block_ids)),
            )
            for group_id, block_ids in zip(group_ids, block_groups)
            if group_id not in self._boundary_handoff_groups
        }
        block_ranges = {
            group_id: tuple(
                index
                for relative_index, index in enumerate(indices)
                if mask is None or mask[relative_index]
            )
            for group_id, indices in candidate_ranges.items()
            for mask in (self._store_mask(request, group_id, indices),)
        }
        resident_blocks = (
            self._restored_store_blocks(
                request, block_ranges, tracker.load_spec.external_tokens, block_groups
            )
            if tracker.load_spec is not None
            else {}
        )
        for group_id, block_ids in zip(group_ids, block_groups):
            if group_id not in block_ranges:
                continue
            group_block_size = self.group_block_sizes[group_id]
            resident = resident_blocks.get(group_id, ())
            for index in block_ranges[group_id]:
                if block_ids[index] == NULL_BLOCK_ID or index in resident:
                    continue
                token_end = (index + 1) * group_block_size
                block_hash = self._object_hash_at_token_end(
                    request.block_hashes, token_end
                )
                key = self.codec.key(block_hash, group_id=group_id)
                if skip_keys is not None and key in skip_keys:
                    continue
                keys.append(key)
                source_block_ids.append(block_ids[index])
                plan_group_ids.append(group_id)
                block_hashes.append(block_hash)
                parent_block_hashes.append(
                    self._object_hash_at_token_end(
                        request.block_hashes, token_end - group_block_size
                    )
                    if self.enable_kv_cache_events and index > 0
                    else None
                )
                token_ids_by_block.append(
                    tuple(request.all_token_ids[index * group_block_size : token_end])
                    if self.enable_kv_cache_events
                    else ()
                )
                block_sizes.append(group_block_size)
        return BlockStoreBatch(
            keys,
            source_block_ids,
            plan_group_ids,
            block_hashes,
            parent_block_hashes,
            token_ids_by_block,
            block_sizes,
            request_id,
        )

    def _restored_store_blocks(
        self,
        request: Any,
        block_ranges: Mapping[int, Sequence[int]],
        restored_tokens: int,
        block_groups: tuple[list[int], ...],
    ) -> dict[int, set[int]]:
        """Query residency before constructing plans for restored blocks."""
        assert self.runtime is not None and self.codec is not None
        # block_ranges covers only positionally stored groups; block_groups
        # follows prefix_cacheable_group_ids.
        blocks_of = dict(
            zip(
                self.kv_cache_config.prefix_cacheable_group_ids,
                block_groups,
                strict=True,
            )
        )
        candidates = {
            group_id: [
                index
                for index in indices
                if index < restored_tokens // self.group_block_sizes[group_id]
                if blocks_of[group_id][index] != NULL_BLOCK_ID
            ]
            for group_id, indices in block_ranges.items()
        }
        keys = [
            key
            for group_id, indices in candidates.items()
            for index in indices
            for key in _rank_keys(
                self.codec,
                self.topology,
                self._object_hash_at_token_end(
                    request.block_hashes, (index + 1) * self.group_block_sizes[group_id]
                ),
                group_id,
            )
        ]
        if not keys:
            return {}
        try:
            hits = list(self.runtime.lookup(keys))
        except Exception as exc:
            logger.debug("UMBP store residency check failed: %s", exc)
            return {}
        if len(hits) != len(keys):
            return {}
        ranks = self.topology.rank_count
        offset = 0
        resident = {}
        for group_id, indices in candidates.items():
            resident[group_id] = {
                index
                for position, index in enumerate(indices)
                if all(
                    hits[offset + position * ranks : offset + (position + 1) * ranks]
                )
            }
            offset += len(indices) * ranks
        return resident

    def _store_mask(
        self,
        request: Any,
        group_id: int,
        indices: range,
    ) -> list[bool] | None:
        """Keep only blocks that core's cache semantics can later reuse."""
        if not indices:
            return []
        group = self.kv_cache_config.kv_cache_groups[group_id]
        spec = _unwrap_store_spec(group.kv_cache_spec)
        manager_cls = KVCacheSpecRegistry.get_manager_class(spec)
        if manager_cls is None:
            return None
        num_prompt_tokens = getattr(request, "num_prompt_tokens", None)
        reachable_boundaries = (
            ()
            if num_prompt_tokens is None
            else replay_boundaries(
                num_prompt_tokens, self.block_size, use_eagle=self.use_eagle
            )
        )
        mask = manager_cls.reachable_block_mask(
            start_block=indices.start,
            end_block=indices.stop,
            alignment_tokens=self.block_size,
            kv_cache_spec=spec,
            use_eagle=group_id in self._eagle_store_groups,
            retention_interval=self._retention_interval,
            reachable_boundaries=reachable_boundaries,
            dcp_world_size=self._dcp_size,
        )
        if mask is not None and len(mask) != len(indices):
            raise RuntimeError(
                f"UMBP store mask for group {group_id} returned {len(mask)} "
                f"entries for {len(indices)} blocks"
            )
        return mask

    def _log_hit(self, request: Request, num_external_tokens: int) -> None:
        request_id = request.request_id
        if request_id in self._hit_log_done:
            return  # a preempted request is re-admitted; log its first admission
        self._hit_log_done.add(request_id)
        local = self._hit_log_local.pop(request_id, 0)
        hashes = list(request.block_hashes)
        seen = self._hit_log_seen
        shared = 0
        for block_hash in hashes:
            if block_hash not in seen:
                break
            shared += 1
        prompt = request.num_prompt_tokens
        theo = min(shared * self.hash_block_size, max(prompt - 1, 0))
        # Seconds since the first request carrying the block that ends at the
        # aligned boundary below theo (and the one ending at theo) finished;
        # -1 while it is still running (its state is not cached yet).
        now = time.monotonic()
        ages = []
        for end in (theo // self.block_size * self.block_size, theo):
            index = end // self.hash_block_size - 1
            if end <= 0 or index >= shared:
                ages.append(-2.0)
                continue
            done = self._hit_log_finished.get(seen[hashes[index]])
            ages.append(-1.0 if done is None else now - done)
        for block_hash in hashes:
            seen.setdefault(block_hash, request_id)
        logger.info(
            "UMBP hit req=%s prompt=%d local=%d external=%d theo=%d unit=%d "
            "block=%d hashes=%d src_age_aligned=%.1f src_age_theo=%.1f",
            request_id,
            prompt,
            local,
            num_external_tokens,
            theo,
            self.hash_block_size,
            self.block_size,
            len(hashes),
            ages[0],
            ages[1],
        )

    def request_finished(
        self, request: Request, block_ids: tuple[list[int], ...]
    ) -> tuple[bool, dict[str, Any] | None]:
        if self._pf_enabled:
            self._pf_drop(request.request_id)
        if self._chain_touch:
            try:
                self._chain_touch_finished(request)
            except Exception as exc:  # noqa: BLE001
                self._chain_touch = False
                logger.exception("UMBP-CT disabled after an error: %r", exc)
        self._hit_log_done.discard(request.request_id)
        if self._hit_log:
            self._hit_log_finished.setdefault(request.request_id, time.monotonic())
        self._hit_log_local.pop(request.request_id, None)
        self._pending_loads.pop(request.request_id, None)
        self._load_specs.pop(request.request_id, None)
        self._requests.pop(request.request_id, None)
        pending = self._pending_lookups.pop(request.request_id, None)
        if pending is not None:
            pending.future.cancel()
        return False, None

    def register_finished_partial_tail(
        self,
        request: Request,
        block_ids: tuple[list[int], ...],
        partial_tail_offloads: list[tuple[int, int, int]],
    ) -> bool:
        """Pin finished boundary pages without delaying core request cleanup."""
        pool = self._gpu_block_pool
        if (
            not self.store_enabled
            or self.codec is None
            or not partial_tail_offloads
            or pool is None
        ):
            return False
        if request.request_id in self._pending_finished_stores:
            return False
        prefix_groups = set(self.kv_cache_config.prefix_cacheable_group_ids)
        tracker = self._request_trackers.get(request.request_id)
        if tracker is None:
            return False
        boundaries = {entry[2] for entry in partial_tail_offloads}
        if len(boundaries) != 1:
            logger.warning("UMBP partial tail entries span several boundaries")
            return False
        boundary = boundaries.pop()
        if (
            boundary <= 0
            or boundary % self.hash_block_size != 0
            or boundary != request.num_computed_tokens
            or bool(getattr(request, "num_in_flight_tokens", 0))
        ):
            return False
        if boundary // self.hash_block_size > len(request.block_hashes):
            return False
        sources = {
            group_id: block_id for group_id, block_id, _ in partial_tail_offloads
        }
        if not sources.keys() <= prefix_groups:
            return False
        # Core hands off Mamba state pages. Include the matching FA tail page
        # so every group can resolve the same fine-grained prefix boundary.
        key_ends = dict.fromkeys(sources, boundary)
        for group_id in prefix_groups:
            if group_id not in self._attention_tail_groups:
                continue
            end = self._attention_tail_end(group_id, boundary)
            group_block_size = self.group_block_sizes[group_id]
            if not end % group_block_size:
                continue
            index = end // group_block_size
            if index >= len(block_ids[group_id]) or end // self.hash_block_size > len(
                request.block_hashes
            ):
                return False
            sources.setdefault(group_id, block_ids[group_id][index])
            key_ends.setdefault(group_id, end)
        plans: list[BlockTransferPlan] = []
        for group_id, block_id in sources.items():
            if block_id == NULL_BLOCK_ID or not 0 <= block_id < len(pool.blocks):
                return False
            if pool.blocks[block_id].is_null:
                return False
            block_hash = request.block_hashes[
                key_ends[group_id] // self.hash_block_size - 1
            ]
            plans.append(
                BlockTransferPlan(
                    request_id=request.request_id,
                    block_id=block_id,
                    group_id=group_id,
                    key=self.codec.key(block_hash, group_id),
                    block_hash=block_hash,
                    block_size=self.group_block_sizes[group_id],
                )
            )
        # A Mamba page is a state snapshot, not a token-addressable byte array.
        # Store full pages for both state and attention, matching restore.
        pool.touch(
            [pool.blocks[block_id] for block_id in dict.fromkeys(sources.values())]
        )
        self._pending_finished_stores[request.request_id] = plans
        return False

    def update_connector_output(self, output: KVConnectorOutput) -> None:
        metadata = output.kv_connector_worker_meta
        if not isinstance(metadata, UMBPConnectorWorkerMetadata):
            return
        pool = self._gpu_block_pool
        for event_id, result in metadata.store_events.items():
            event = self._store_events.get(event_id)
            if event is None:
                continue
            event.result.merge(result)
            if event.result.completed_workers < self._num_workers:
                continue
            del self._store_events[event_id]
            if pool is not None:
                block_ids = dict.fromkeys(event.block_ids())
                pool.free_blocks(
                    pool.blocks[block_id] for block_id in reversed(block_ids)
                )
            if self.enable_kv_cache_events:
                self._queue_event_objects(event)
        self._reconcile_kv_events()

    def _queue_event_objects(self, event: _StoreEvent) -> None:
        seen: set[tuple[int, bytes]] = set()
        for plan in event.iter_plans():
            if plan.group_id is None or plan.block_hash is None:
                continue
            identity = (plan.group_id, plan.block_hash)
            if identity in seen or identity in event.result.failed_objects:
                continue
            seen.add(identity)
            previous = self._event_objects.get(identity)
            if previous is None and len(self._event_objects) >= self._event_max_objects:
                self._event_publications_suppressed += 1
                logger.warning_once(
                    "UMBP KV event tracking limit reached; suppressing new "
                    "BlockStored events until tracked objects are evicted"
                )
                continue
            self._event_serial += 1
            self._event_objects[identity] = _EventObject(
                self._event_serial, plan, previous.published if previous else False
            )
            self._pending_event_objects[identity] = self._event_serial

    def invalidate_kv_events(self, events: Sequence[KVCacheEvent]) -> None:
        removed = {
            (event.group_idx, block_hash)
            for event in events
            if isinstance(event, BlockRemoved)
            for block_hash in event.block_hashes
        }
        if not removed:
            return
        for identity in list(self._event_objects):
            group_id, block_hash = identity
            if (
                group_id,
                maybe_convert_block_hash(cast(BlockHash, block_hash)),
            ) in removed:
                del self._event_objects[identity]
                self._pending_event_objects.pop(identity, None)

    def _reconcile_kv_events(self) -> None:
        if (
            not self.enable_kv_cache_events
            or self.runtime is None
            or self.codec is None
        ):
            return
        pending = self._event_lookup
        if pending is not None:
            if not pending.future.done():
                return
            self._event_lookup = None
            try:
                hits = list(pending.future.result())
                if len(hits) != len(pending.objects) * self.topology.rank_count:
                    raise ValueError("invalid UMBP event residency result length")
            except Exception as exc:
                logger.debug("UMBP event residency check failed: %s", exc)
                hits = None
            for index, (identity, serial) in enumerate(pending.objects):
                record = self._event_objects.get(identity)
                if record is None or record.serial != serial:
                    continue
                # Rotate completed checks so a bounded batch visits every object.
                del self._event_objects[identity]
                self._pending_event_objects.pop(identity, None)
                if hits is None:
                    if record.published:
                        self._event_objects[identity] = record
                    continue
                ranks = self.topology.rank_count
                resident = all(hits[index * ranks : (index + 1) * ranks])
                if not resident:
                    if record.published:
                        group_id, block_hash = identity
                        self._kv_events.append(
                            BlockRemoved(
                                block_hashes=[
                                    maybe_convert_block_hash(
                                        cast(BlockHash, block_hash)
                                    )
                                ],
                                medium="CPU",
                                group_idx=group_id,
                                locality="LOCAL",
                            )
                        )
                    continue
                if not record.published:
                    plan = record.plan
                    assert plan is not None and plan.block_hash is not None
                    self._kv_events.append(
                        BlockStored(
                            block_hashes=[
                                maybe_convert_block_hash(
                                    cast(BlockHash, plan.block_hash)
                                )
                            ],
                            parent_block_hash=(
                                maybe_convert_block_hash(
                                    cast(BlockHash, plan.parent_block_hash)
                                )
                                if plan.parent_block_hash is not None
                                else None
                            ),
                            token_ids=list(plan.token_ids),
                            block_size=plan.block_size,
                            lora_id=None,
                            medium=plan.medium,
                            lora_name=None,
                            group_idx=plan.group_id,
                            locality="LOCAL",
                        )
                    )
                record.published = True
                record.plan = None
                self._event_objects[identity] = record
        if not self._event_objects:
            return
        now = time.monotonic()
        candidates = []
        if now >= self._next_event_reconcile:
            candidates = list(
                islice(
                    (
                        (identity, record.serial)
                        for identity, record in self._event_objects.items()
                        if record.published
                    ),
                    max(1, self._event_check_batch_size // 2),
                )
            )
            self._next_event_reconcile = now + 1.0
        selected = {identity for identity, _ in candidates}
        candidates.extend(
            islice(
                (
                    (identity, serial)
                    for identity, serial in self._pending_event_objects.items()
                    if identity not in selected
                ),
                self._event_check_batch_size - len(candidates),
            )
        )
        if not candidates:
            return
        keys = [
            key
            for (group_id, block_hash), _ in candidates
            for key in _rank_keys(self.codec, self.topology, block_hash, group_id)
        ]
        self._event_lookup = _EventLookup(
            tuple(candidates),
            self._lookup_executor.submit(self.runtime.lookup, keys),
        )

    def take_events(self) -> list[KVCacheEvent]:
        self._reconcile_kv_events()
        events = self._kv_events
        self._kv_events = []
        return events

    def has_pending_push_work(self) -> bool:
        return (
            bool(self._pending_finished_stores)
            or bool(self._store_events)
            or self._event_lookup is not None
            or bool(self._pending_event_objects)
        )

    def reset_store(self) -> bool:
        if self._store_events or self._pending_finished_stores:
            return False
        for pending in self._pending_lookups.values():
            pending.future.cancel()
        self._pending_lookups.clear()
        self._load_specs.clear()
        self._pending_loads.clear()
        cleared = True if self.runtime is None else self.runtime.clear()
        if cleared:
            self._event_objects.clear()
            self._pending_event_objects.clear()
            if self._event_lookup is not None:
                self._event_lookup.future.cancel()
                self._event_lookup = None
        return cleared

    def close(self) -> None:
        self._lookup_executor.shutdown(wait=True, cancel_futures=True)
        if self._pf_executor is not None:
            self._pf_executor.shutdown(wait=False, cancel_futures=True)
        self._pending_lookups.clear()
        if self.runtime is not None:
            self.runtime.close()


# ---- prefix-hit debug wrappers (image patch, VLLM_PREFIX_HIT_DEBUG=1) ----
import os as _phd_os

if _phd_os.environ.get("VLLM_PREFIX_HIT_DEBUG") == "1":
    from vllm.logger import init_logger as _phd_init_logger

    _phd_logger = _phd_init_logger("vllm.prefix_hit_debug")

    _phd_orig_reg = UMBPStoreConnectorScheduler.register_finished_partial_tail
    _phd_orig_bsp = UMBPStoreConnectorScheduler._add_boundary_state_plans

    def _phd_reg(self, request, block_ids, partial_tail_offloads):
        accepted = _phd_orig_reg(self, request, block_ids, partial_tail_offloads)
        _phd_logger.info(
            "PHD tailreg req=%s computed=%d boundaries=%s accepted=%s",
            request.request_id, request.num_computed_tokens,
            sorted({entry[2] for entry in partial_tail_offloads}), accepted,
        )
        return accepted

    def _phd_bsp(self, offloads, meta, finished, preempted):
        for request_id, entries in offloads.items():
            _phd_logger.info(
                "PHD bstore req=%s boundaries=%s finished=%s",
                request_id, sorted({entry[2] for entry in entries}),
                request_id in finished,
            )
        return _phd_orig_bsp(self, offloads, meta, finished, preempted)

    _phd_orig_gnnmt = UMBPStoreConnectorScheduler.get_num_new_matched_tokens

    def _phd_gnnmt(self, request, num_computed_tokens):
        self._phd_cur_req = request.request_id
        return _phd_orig_gnnmt(self, request, num_computed_tokens)

    _phd_orig_coord = UMBPStoreConnectorScheduler._coordinate_external_hits

    def _phd_coord(self, request_hashes, local_tokens, max_external_tokens,
                   logical_objects_by_group, logical_hits_by_group):
        result = _phd_orig_coord(self, request_hashes, local_tokens, max_external_tokens,
                                 logical_objects_by_group, logical_hits_by_group)
        try:
            parts = []
            for gid, hits in logical_hits_by_group.items():
                unit = self._lookup_units[gid]
                ends = [local_tokens + (i + 1) * unit for i, hit in enumerate(hits) if hit]
                block = self.group_block_sizes[gid]
                run = 0
                for i in range(block // unit - 1, len(hits), block // unit):
                    if not hits[i]:
                        break
                    run = local_tokens + (i + 1) * unit
                parts.append("g%d:obj=%d,hit=%d,max=%s,blockrun=%d,last=%s" % (
                    gid, len(hits), len(ends), ends[-1] if ends else 0, run, ends[-3:]))
            _phd_logger.info(
                "PHD lookup req=%s local=%d max_ext=%d load=%d %s",
                getattr(self, "_phd_cur_req", None), local_tokens, max_external_tokens,
                result[0], " ".join(parts),
            )
        except Exception as exc:  # debug only
            _phd_logger.info("PHD lookup log failed: %r", exc)
        return result

    UMBPStoreConnectorScheduler.register_finished_partial_tail = _phd_reg
    UMBPStoreConnectorScheduler._add_boundary_state_plans = _phd_bsp
    UMBPStoreConnectorScheduler._coordinate_external_hits = _phd_coord
    UMBPStoreConnectorScheduler.get_num_new_matched_tokens = _phd_gnnmt
