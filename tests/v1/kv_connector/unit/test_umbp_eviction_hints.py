# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Lookup fan-out, recency hints and eviction classes of the UMBP connector."""

from types import SimpleNamespace

import pytest

from tests.v1.kv_connector.umbp_test_utils import (
    _hybrid_kv_cache_config,
    _kv_cache_config,
    _SchedulerHandle,
    _vllm_config,
    _WorkerHandle,
    make_request,
    make_scheduler_output,
)
from vllm.distributed.kv_transfer.kv_connector.v1.umbp.data import (
    BlockIdentityCodec,
    BlockStoreBatch,
    BlockTransferPlan,
    RankTopology,
    TransferJobState,
    UMBPNamespace,
)
from vllm.distributed.kv_transfer.kv_connector.v1.umbp.runtime.embedded import (
    _lookup_socket_path,
    _MoriLookupServer,
    _MoriSchedulerHandle,
    _rank_namespace_from_key_prefix,
)
from vllm.distributed.kv_transfer.kv_connector.v1.umbp.scheduler import (
    UMBPStoreConnectorScheduler,
)
from vllm.distributed.kv_transfer.kv_connector.v1.umbp.worker import (
    UMBPStoreConnectorWorker,
)


class _TouchRecordingHandle(_SchedulerHandle):
    def __init__(self, hits):
        super().__init__(hits)
        self.touched = []

    def touch(self, keys):
        self.touched.append(list(keys))


def _scheduler(extra, *, prefix_match_unit=None, tp_size=1, hybrid=True, handle=None):
    topology = RankTopology(tp_size=tp_size)
    codec = BlockIdentityCodec(UMBPNamespace("hints"))
    scheduler = UMBPStoreConnectorScheduler(
        _vllm_config(
            {"mode": "embedded", **extra},
            prefix_match_unit=prefix_match_unit,
            tensor_parallel_size=tp_size,
            world_size=tp_size,
        ),
        _hybrid_kv_cache_config() if hybrid else _kv_cache_config(),
        handle if handle is not None else _TouchRecordingHandle({}),
        codec,
        topology,
    )
    return scheduler, codec, topology


@pytest.mark.parametrize(
    ("extra", "prefix_match_unit", "hybrid", "expected"),
    [
        ({}, 4, True, True),
        ({"enable_partial_hash_hits": False}, 4, True, False),
        ({}, None, True, False),
        ({}, None, False, False),
    ],
)
def test_hybrid_models_default_to_hash_granular_lookup(
    extra, prefix_match_unit, hybrid, expected
):
    """Mamba checkpoints sit on hash boundaries, so a hybrid model with a finer
    prefix_match_unit needs hash-granular lookup unless told otherwise."""
    scheduler, _, _ = _scheduler(
        extra, prefix_match_unit=prefix_match_unit, hybrid=hybrid
    )
    try:
        assert scheduler.enable_partial_hash_hits is expected
    finally:
        scheduler.close()


def test_lookup_asks_other_ranks_only_about_what_the_first_rank_holds():
    handle = _TouchRecordingHandle({})
    scheduler, codec, topology = _scheduler({}, tp_size=2, handle=handle)
    try:
        keys = [
            key
            for block_hash in (b"a", b"b", b"c")
            for key in codec.keys_for_topology(block_hash, topology, (0,))
        ]
        a0, a1, b0, b1, c0, c1 = keys
        # a: both ranks; b: first rank only; c: second rank only.
        handle.hits = {a0: True, a1: True, b0: True, c1: True}

        hits = scheduler._lookup_all_ranks(keys)

        assert handle.queries == [[a0, b0, c0], [a1, b1]]
        # Per object, every rank holds it only for a, as one full query says.
        assert [all(hits[i : i + 2]) for i in range(0, 6, 2)] == [True, False, False]
        handle.queries.clear()
        handle.hits = {}
        assert scheduler._lookup_all_ranks(keys) == [False] * 6
        assert handle.queries == [[a0, b0, c0]]
    finally:
        scheduler.close()


def test_lookup_through_two_phases_matches_the_full_query():
    block_hashes = [bytes([index]) for index in range(8)]
    full = _TouchRecordingHandle({})
    phased = _TouchRecordingHandle({})
    single, codec, topology = _scheduler(
        {"load_async": False}, prefix_match_unit=4, tp_size=2, handle=full
    )
    split, _, _ = _scheduler(
        {"load_async": False}, prefix_match_unit=4, tp_size=2, handle=phased
    )
    try:
        stored = {
            key
            for group in (0, 1)
            for block_hash in block_hashes[:5]
            for key in codec.keys_for_topology(block_hash, topology, (group,))
        }
        full.hits = phased.hits = dict.fromkeys(stored, True)
        # The reference asks every rank about every object in one query.
        single._lookup_all_ranks = lambda keys: list(full.lookup(keys))
        request = make_request("phased", num_tokens=33, block_hashes=block_hashes)
        expected = single.get_num_new_matched_tokens(request, 0)
        assert expected[0] > 0
        assert split.get_num_new_matched_tokens(request, 0) == expected
        # The first query names only the first rank's copies.
        assert all(":tp0:" in key for key in phased.queries[0])
    finally:
        single.close()
        split.close()


def test_local_hit_touches_every_rank_copy_of_its_objects():
    handle = _TouchRecordingHandle({})
    scheduler, codec, topology = _scheduler(
        {"load_async": False}, prefix_match_unit=4, tp_size=2, handle=handle
    )
    block_hashes = [bytes([index]) for index in range(8)]
    request = make_request(
        "hit",
        num_tokens=33,
        block_hashes=block_hashes,
        block_ids=([1, 2, 3], [4, 5, 6]),
    )
    scheduler.update_state_after_alloc(
        request, SimpleNamespace(get_block_ids=lambda group_ids: request.block_ids), 0
    )
    output = make_scheduler_output(
        new_reqs=[
            SimpleNamespace(
                req_id="hit", num_computed_tokens=20, block_ids=request.block_ids
            )
        ],
        scheduled_tokens={"hit": 13},
    )
    scheduler.build_connector_meta(output)
    scheduler.close()  # waits for the background touch

    def keys(group, *hash_indices):
        return {
            key
            for index in hash_indices
            for key in codec.keys_for_topology(block_hashes[index], topology, (group,))
        }

    # Attention: the full first block (tokens 16) and the tail-page candidates at
    # 20/24/28; Mamba: the checkpoint the hit resumes from, at 20.
    expected = keys(0, 3, 4, 5, 6) | keys(1, 4)
    assert len(handle.touched) == 1
    assert set(handle.touched[0]) == expected


def test_no_local_hit_sends_no_touch():
    handle = _TouchRecordingHandle({})
    scheduler, _, _ = _scheduler(
        {"load_async": False}, prefix_match_unit=4, handle=handle
    )
    request = make_request("miss", num_tokens=33, block_hashes=[b"a"] * 8)
    scheduler.update_state_after_alloc(
        request, SimpleNamespace(get_block_ids=lambda group_ids: ([1], [2])), 0
    )
    scheduler.build_connector_meta(
        make_scheduler_output(
            new_reqs=[
                SimpleNamespace(
                    req_id="miss", num_computed_tokens=0, block_ids=([1], [2])
                )
            ],
            scheduled_tokens={"miss": 33},
        )
    )
    scheduler.close()
    assert handle.touched == []


class _ClassRecordingWorkerHandle(_WorkerHandle):
    def __init__(self):
        self.classes = []

    def set_eviction_class(self, keys, eviction_class):
        self.classes.append((list(keys), eviction_class))


@pytest.mark.parametrize("columnar", [False, True])
def test_worker_marks_only_state_objects_low_priority(columnar):
    handle = _ClassRecordingWorkerHandle()
    worker = UMBPStoreConnectorWorker(handle, low_priority_group_ids=frozenset({1}))
    if columnar:
        plans = BlockStoreBatch(
            keys=["kv", "state-a", "state-b"],
            block_ids=[1, 2, 3],
            group_ids=[0, 1, 1],
            block_hashes=[b"x", b"y", b"z"],
            parent_block_hashes=[None, None, None],
            token_ids=[(), (), ()],
            block_sizes=[16, 16, 16],
            request_id="r",
        )
    else:
        plans = tuple(
            BlockTransferPlan(key=key, block_id=index, group_id=group)
            for index, (key, group) in enumerate(
                (("kv", 0), ("state-a", 1), ("state-b", 1))
            )
        )
    job = TransferJobState(plans)
    job.start()
    job.complete()

    assert worker._finish_job("r", job, is_load=False)
    assert handle.classes == [(["state-a", "state-b"], 1)]


def test_worker_without_class_support_or_state_groups_marks_nothing():
    plans = (BlockTransferPlan(key="state", block_id=1, group_id=1),)
    for handle, groups in (
        (_WorkerHandle(), frozenset({1})),
        (_ClassRecordingWorkerHandle(), frozenset()),
    ):
        worker = UMBPStoreConnectorWorker(handle, low_priority_group_ids=groups)
        job = TransferJobState(plans)
        job.start()
        job.complete()
        assert worker._finish_job("r", job, is_load=False)
        assert getattr(handle, "classes", []) == []


def test_embedded_touch_reaches_each_ranks_pool(tmp_path):
    _rank_namespace_from_key_prefix.cache_clear()

    class _Client:
        def __init__(self):
            self.touched = []

        def batch_exists(self, keys):
            return [False] * len(keys)

        def batch_touch(self, keys):
            self.touched.append(list(keys))
            return len(keys)

        def clear(self):
            return True

    namespace = "touch-routing"
    topology = RankTopology(tp_size=2)
    clients = [_Client(), _Client()]
    servers = []
    for rank, client in zip(topology.all_namespaces(), clients, strict=True):
        server = _MoriLookupServer(
            _lookup_socket_path(namespace, rank, str(tmp_path)), client
        )
        server.start()
        servers.append(server)
    scheduler = _MoriSchedulerHandle(namespace, topology, str(tmp_path))
    codec = BlockIdentityCodec(UMBPNamespace(namespace))
    try:
        keys = list(codec.keys_for_topology(b"a", topology, (0, 1)))
        scheduler.touch(keys)
        for rank, client in enumerate(clients):
            assert client.touched == [[key for key in keys if f":tp{rank}:" in key]]
    finally:
        for server in servers:
            server.close()
