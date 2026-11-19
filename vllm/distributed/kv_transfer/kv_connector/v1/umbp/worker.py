# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared worker-side UMBP connector behavior."""

from __future__ import annotations

import os
import time
from collections import Counter
from dataclasses import dataclass, field, replace
from typing import cast

import torch

from vllm.distributed.kv_events import BlockRemoved, KVCacheEvent
from vllm.forward_context import ForwardContext
from vllm.logger import init_logger
from vllm.v1.core.kv_cache_utils import BlockHash, maybe_convert_block_hash
from vllm.v1.kv_cache_interface import mamba_tail_window

from .data import (
    BlockIdentityCodec,
    BlockLoadBatch,
    BlockStoreBatch,
    BlockTransferPlan,
    KVLayoutPlanner,
    StoreEventResult,
    TransferJobState,
    TransferJobStatus,
    UMBPConnectorMetadata,
    UMBPConnectorWorkerMetadata,
)
from .runtime import UMBPWorkerHandle
from .stats import UMBPStoreConnectorStats

logger = init_logger(__name__)


@dataclass
class _StoreBatch:
    event_id: int
    jobs: dict[str, TransferJobState]
    failed_objects: set[tuple[int, bytes]] = field(default_factory=set)


class UMBPStoreConnectorWorker:
    """Translate shared metadata into runtime load/store jobs."""

    def __init__(
        self,
        runtime: UMBPWorkerHandle,
        layout: KVLayoutPlanner | None = None,
        *,
        codec: BlockIdentityCodec | None = None,
        report_failed_requests: bool = False,
        enable_kv_cache_events: bool = False,
        low_priority_group_ids: frozenset[int] = frozenset(),
    ) -> None:
        self.runtime = runtime
        self.layout = layout
        self.codec = codec
        self.enable_kv_cache_events = enable_kv_cache_events
        # Groups whose objects the pool should evict sooner: recurrent-state
        # checkpoints are stored for every prompt but rarely read back.
        self._low_priority_group_ids = low_priority_group_ids
        self._load_jobs: dict[str, dict[str | None, TransferJobState]] = {}
        self._store_jobs: dict[str, TransferJobState] = {}
        self._pending_stores: list[_StoreBatch] = []
        self._worker_meta = UMBPConnectorWorkerMetadata()
        self._kv_events: list[KVCacheEvent] = []
        self._finished_recving: set[str] = set()
        # Core maps block-level load failures to requests only with one group.
        self._report_failed_requests = report_failed_requests
        self._failed_recving: set[str] = set()
        self._load_error_block_ids: set[int] = set()
        self._report_load_completions = False
        self._active_store_event = -1
        self._stats = UMBPStoreConnectorStats()
        # PHD wstore: rank-0 store-job completion log (debug overlay).
        self._phd = os.environ.get("VLLM_PREFIX_HIT_DEBUG") == "1" and (
            codec is None or codec.tp_rank == 0
        )
        self._phd_t: dict[int, tuple[float, str]] = {}
        # K3WSTEP (VLLM_UMBP_STEP_LOG=1, rank 0): how long this worker's forward passes
        # take and how long it sits between them, one line per interval.
        self._wstep = os.environ.get("VLLM_UMBP_STEP_LOG") == "1" and (
            codec is None or codec.tp_rank == 0
        )
        self._wstep_t0 = time.monotonic()
        self._wstep_begin = 0.0
        self._wstep_end = 0.0
        self._wstep_acc = {"steps": 0, "fwd_s": 0.0, "idle_s": 0.0, "hook_s": 0.0,
                           "loads": 0, "load_keys": 0, "idle_max": 0.0, "fwd_max": 0.0}

    def _wstep_start(self) -> None:
        now = time.monotonic()
        acc = self._wstep_acc
        if self._wstep_end:
            idle = now - self._wstep_end
            acc["idle_s"] += idle
            acc["idle_max"] = max(acc["idle_max"], idle)
        self._wstep_begin = now

    def _wstep_finish(self, hook_started: float) -> None:
        now = time.monotonic()
        acc = self._wstep_acc
        if self._wstep_begin:
            fwd = hook_started - self._wstep_begin
            acc["steps"] += 1
            acc["fwd_s"] += fwd
            acc["fwd_max"] = max(acc["fwd_max"], fwd)
        acc["hook_s"] += now - hook_started
        self._wstep_end = now
        self._wstep_begin = 0.0
        if now - self._wstep_t0 >= 10.0:
            logger.info(
                "K3WSTEP dt=%.1f steps=%d fwd_s=%.3f idle_s=%.3f hook_s=%.3f "
                "fwd_max=%.3f idle_max=%.3f loads=%d load_keys=%d",
                now - self._wstep_t0, acc["steps"], acc["fwd_s"], acc["idle_s"],
                acc["hook_s"], acc["fwd_max"], acc["idle_max"], acc["loads"],
                acc["load_keys"],
            )
            self._wstep_t0 = now
            for key in acc:
                acc[key] = 0 if isinstance(acc[key], int) else 0.0

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]) -> None:
        if self.layout is not None and self.layout.regions:
            self.layout.register_kv_caches(kv_caches)
        self.runtime.register_buffers(kv_caches)

    def _materialize_plans(
        self, plans: list[BlockTransferPlan]
    ) -> list[BlockTransferPlan]:
        plans = [self._localize_plan(plan) for plan in plans]
        if self.layout is None or not self.layout.regions:
            return plans
        materialized: list[BlockTransferPlan] = []
        for plan in plans:
            if plan.ranges:
                materialized.append(plan)
                continue
            materialized.append(self.layout.materialize(plan))
        return materialized

    def _localize_plan(self, plan: BlockTransferPlan) -> BlockTransferPlan:
        if self.codec is None or plan.group_id is None:
            return plan
        key = self._localize_key(plan.key, plan.group_id)
        if key == plan.key:
            return plan
        return replace(plan, key=key)

    def _localize_key(self, key: str, group_id: int) -> str:
        if self.codec is None:
            return key
        try:
            block_hash = bytes.fromhex(key.rsplit(":", 1)[1])
        except (IndexError, ValueError):
            return key
        return self.codec.key(block_hash, group_id)

    def start_load_kv(
        self, forward_context: ForwardContext, metadata: UMBPConnectorMetadata
    ) -> None:
        del forward_context
        if self._wstep:
            self._wstep_start()
        self._report_load_completions = metadata.async_load
        for request_id, plans in metadata.load_requests.items():
            if plans:
                self._submit_loads(request_id, plans)
                if self._wstep:
                    self._wstep_acc["loads"] += 1
                    self._wstep_acc["load_keys"] += len(plans)
        if not metadata.async_load:
            self._drain_load_jobs(wait=True)

    def _submit_loads(
        self, request_id: str, plans: list[BlockTransferPlan] | BlockLoadBatch
    ) -> None:
        load_blocks = getattr(self.runtime, "load_blocks", None)
        if load_blocks is not None:
            localized: list[BlockTransferPlan] | BlockLoadBatch
            if isinstance(plans, BlockLoadBatch):
                localized = plans
                if self.codec is not None:
                    localized = replace(
                        plans,
                        keys=[
                            self._localize_key(key, group)
                            for key, group in zip(
                                plans.keys, plans.group_ids, strict=True
                            )
                        ],
                    )
            else:
                localized = [self._localize_plan(plan) for plan in plans]
            job = load_blocks(localized)
            if job is not None:
                self._stats.record("load", submitted=len(plans))
                self._load_jobs[request_id] = {None: job}
                if self._phd:
                    self._phd_t[id(job)] = (time.monotonic(), "load")
                return
        materialized = self._materialize_plans(list(plans))
        logger.debug(
            "UMBP load submitted request=%s plans=%d ranges=%d bytes=%d",
            request_id,
            len(materialized),
            sum(len(plan.ranges) for plan in materialized),
            sum(item.length for plan in materialized for item in plan.ranges),
        )
        self._stats.record("load", submitted=len(materialized))
        self._load_jobs[request_id] = {None: self.runtime.load(materialized)}

    def _finish_job(
        self, request_id: str, result: TransferJobState, *, is_load: bool
    ) -> bool:
        operation = "load" if is_load else "store"
        if self._phd and is_load:
            t0, _ = self._phd_t.pop(id(result), (None, "?"))
            logger.info(
                "PHD wload req=%s keys=%d done=%d failed=%d bytes=%d lat=%s status=%s",
                request_id, len(result.keys), len(result.completed_keys),
                len(result.failed_keys), result.completed_bytes,
                "%.3f" % (time.monotonic() - t0) if t0 is not None else "?",
                result.status.value,
            )
        if self._phd and not is_load:
            t0, groups = self._phd_t.pop(id(result), (None, "?"))
            logger.info(
                "PHD wstore req=%s keys=%d done=%d failed=%d groups=%s lat=%s status=%s",
                request_id, len(result.keys), len(result.completed_keys),
                len(result.failed_keys), groups,
                "%.3f" % (time.monotonic() - t0) if t0 is not None else "?",
                result.status.value,
            )
        self._stats.record(
            operation,
            completed=len(result.completed_keys),
            failed=len(result.failed_keys),
            num_bytes=result.completed_bytes,
        )
        succeeded = result.status == TransferJobStatus.COMPLETED
        if not succeeded:
            logger.debug(
                "UMBP %s failed request=%s status=%s error=%s",
                operation,
                request_id,
                result.status.value,
                result.error,
            )
        if is_load:
            if not self._report_failed_requests:
                self._load_error_block_ids.update(result.failed_block_ids)
            elif not succeeded:
                self._failed_recving.add(request_id)
        elif succeeded:
            self.runtime.publish(result)
            self._mark_low_priority(result)
        return succeeded

    def _mark_low_priority(self, result: TransferJobState) -> None:
        set_class = getattr(self.runtime, "set_eviction_class", None)
        if not callable(set_class):
            return
        plans = result.plans
        if mamba_tail_window() is not None:
            # Window mode: only the plans the scheduler flagged (tail-window
            # states and their attention pages) are low class.
            if isinstance(plans, (BlockLoadBatch, BlockStoreBatch)):
                return
            keys = [
                plan.key
                for plan in plans
                if plan.low_priority and plan.key in result.completed_keys
            ]
            if keys:
                set_class(keys, 1)
            return
        if not self._low_priority_group_ids:
            return
        if isinstance(plans, (BlockLoadBatch, BlockStoreBatch)):
            entries = zip(plans.keys, plans.group_ids, strict=True)
        else:
            entries = ((plan.key, plan.group_id) for plan in plans)
        keys = [
            key
            for key, group_id in entries
            if group_id in self._low_priority_group_ids and key in result.completed_keys
        ]
        if keys:
            set_class(keys, 1)

    def wait_for_layer_load(self, layer_name: str) -> None:
        if self._report_load_completions:
            # Asynchronous loads fill blocks of requests that do not run until
            # get_finished reports them, so no forward pass reads those blocks.
            return
        self._drain_load_jobs(wait=True)

    def wait_for_save(self) -> None:
        hook_started = time.monotonic() if self._wstep else 0.0
        try:
            self._wait_for_save()
        finally:
            if self._wstep:
                self._wstep_finish(hook_started)

    def _wait_for_save(self) -> None:
        if self._store_jobs or self._active_store_event >= 0:
            self._pending_stores.append(
                _StoreBatch(self._active_store_event, self._store_jobs)
            )
            self._store_jobs = {}
            self._active_store_event = -1
        self._drain_store_jobs(wait=False)

    def _drain_store_jobs(self, *, wait: bool) -> None:
        for batch in list(self._pending_stores):
            for request_id, job in list(batch.jobs.items()):
                result = self.runtime.wait(job) if wait else self.runtime.poll(job)
                if result is None:
                    continue
                # A partly failed job is not published.
                if not self._finish_job(request_id, result, is_load=False):
                    batch.failed_objects.update(
                        (plan.group_id, plan.block_hash)
                        for plan in result.plans
                        if plan.group_id is not None and plan.block_hash is not None
                    )
                del batch.jobs[request_id]
            if not batch.jobs:
                if batch.event_id >= 0:
                    self._worker_meta.store_events[batch.event_id] = StoreEventResult(
                        completed_workers=1,
                        failed_objects=batch.failed_objects,
                    )
                self._pending_stores.remove(batch)

    def _submit_store_plans(
        self, plans: list[BlockTransferPlan] | BlockStoreBatch, request_id: str
    ) -> None:
        if not plans:
            return
        store_blocks = getattr(self.runtime, "store_blocks", None)
        if store_blocks is not None:
            localized: list[BlockTransferPlan] | BlockStoreBatch
            if isinstance(plans, BlockStoreBatch):
                localized = plans
                if self.codec is not None:
                    localized = plans.localized(
                        [
                            self._localize_key(key, group_id)
                            for key, group_id in zip(
                                plans.keys, plans.group_ids, strict=True
                            )
                        ]
                    )
            else:
                localized = [self._localize_plan(plan) for plan in plans]
            job = store_blocks(localized)
            if job is not None:
                self._stats.record("store", submitted=len(plans))
                self._phd_note(job, plans)
                self._store_jobs[request_id] = job
                return
        materialized = self._materialize_plans(list(plans))
        self._stats.record(
            "store",
            submitted=len(materialized),
        )
        self._store_jobs[request_id] = self.runtime.store(materialized)
        self._phd_note(self._store_jobs[request_id], materialized)

    def _phd_note(self, job, plans) -> None:
        if not self._phd:
            return
        gids = plans.group_ids if isinstance(plans, BlockStoreBatch) else [p.group_id for p in plans]
        groups = ",".join(f"{g}:{n}" for g, n in sorted(Counter(gids).items()))
        self._phd_t[id(job)] = (time.monotonic(), groups)

    def enqueue_stores(self, metadata: UMBPConnectorMetadata) -> None:
        self._active_store_event = metadata.store_event
        store_requests = dict(metadata.store_requests)
        grouped_plan_ids = {
            id(plan)
            for plans in store_requests.values()
            if not isinstance(plans, BlockStoreBatch)
            for plan in plans
        }
        for plan in metadata.store_plans:
            if id(plan) not in grouped_plan_ids:
                request_id = plan.request_id or "__umbp_batch__"
                if isinstance(store_requests.get(request_id), BlockStoreBatch):
                    request_id = f"__umbp_flat__:{request_id}"
                pending = store_requests.setdefault(request_id, [])
                assert isinstance(pending, list)
                pending.append(plan)
                grouped_plan_ids.add(id(plan))
        for request_id, plans in store_requests.items():
            self._submit_store_plans(plans, request_id)

    def handle_preemptions(self, metadata: UMBPConnectorMetadata) -> None:
        """Cancel request-local jobs before vLLM reuses their GPU blocks."""
        preempted = metadata.preempted_request_ids
        if not preempted:
            return
        for request_id in preempted:
            for job in self._load_jobs.pop(request_id, {}).values():
                self._cancel_job(job)
        for jobs in [self._store_jobs, *(batch.jobs for batch in self._pending_stores)]:
            for request_id, job in jobs.items():
                plan_request_id = (
                    job.plans.request_id
                    if isinstance(job.plans, (BlockLoadBatch, BlockStoreBatch))
                    else None
                )
                if (
                    request_id in preempted
                    or plan_request_id in preempted
                    or (
                        plan_request_id is None
                        and any(plan.request_id in preempted for plan in job.plans)
                    )
                ):
                    jobs[request_id] = self._cancel_job(job)

    def _cancel_job(self, job: TransferJobState) -> TransferJobState:
        cancel = getattr(self.runtime, "cancel", None)
        if callable(cancel):
            return cancel(job)
        return self.runtime.wait(job)

    def get_finished(
        self, finished_req_ids: set[str]
    ) -> tuple[set[str] | None, set[str] | None]:
        del finished_req_ids
        self._drain_load_jobs(wait=False)
        self._drain_store_jobs(wait=False)
        recving = set(self._finished_recving)
        self._finished_recving.difference_update(recving)
        return None, recving or None

    def _drain_load_jobs(self, *, wait: bool) -> None:
        for request_id, jobs in list(self._load_jobs.items()):
            for layer, job in list(jobs.items()):
                result = self.runtime.wait(job) if wait else self.runtime.poll(job)
                if result is None:
                    continue
                del jobs[layer]
                self._finish_job(request_id, result, is_load=True)
            if not jobs:
                del self._load_jobs[request_id]
                if self._report_load_completions:
                    self._finished_recving.add(request_id)

    def get_failed_recving(self) -> set[str]:
        failed = self._failed_recving - self._load_jobs.keys()
        self._failed_recving.difference_update(failed)
        return failed

    def build_connector_worker_meta(self) -> UMBPConnectorWorkerMetadata:
        result = self._worker_meta
        self._worker_meta = UMBPConnectorWorkerMetadata()
        return result

    def get_kv_events(self) -> list[KVCacheEvent]:
        take_evicted = getattr(self.runtime, "take_evicted_keys", None)
        if callable(take_evicted):
            for key in take_evicted():
                if not self.enable_kv_cache_events:
                    continue
                parsed = self._parse_key(key)
                if parsed is None:
                    continue
                group_id, block_hash = parsed
                self._kv_events.append(
                    BlockRemoved(
                        block_hashes=[
                            maybe_convert_block_hash(cast(BlockHash, block_hash))
                        ],
                        medium="CPU",
                        group_idx=group_id,
                        locality="LOCAL",
                    )
                )
        events = self._kv_events
        self._kv_events = []
        return events

    @staticmethod
    def _parse_key(key: str) -> tuple[int, bytes] | None:
        try:
            group_part, hash_hex = key.rsplit(":", 1)
            group_id = int(group_part.rsplit(":g", 1)[1])
            return group_id, bytes.fromhex(hash_hex)
        except (IndexError, ValueError):
            return None

    def get_block_ids_with_load_errors(self) -> set[int]:
        failed = self._load_error_block_ids
        self._load_error_block_ids = set()
        return failed

    def get_kv_connector_stats(self) -> UMBPStoreConnectorStats | None:
        if self._stats.is_empty():
            return None
        result = UMBPStoreConnectorStats(data=dict(self._stats.data))
        self._stats.reset()
        return result

    def close(self) -> None:
        try:
            self.wait_for_save()
            self._drain_store_jobs(wait=True)
            self._drain_load_jobs(wait=True)
        except TimeoutError as exc:
            logger.warning("UMBP transfers are still in flight at shutdown: %s", exc)
        finally:
            self.runtime.close()
