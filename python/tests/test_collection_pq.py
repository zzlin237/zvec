# Copyright 2025-present the zvec project
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""End-to-end tests for PQ quantization (HNSW + IVF, int8/int4)."""

from __future__ import annotations

import numpy as np
import pytest
import zvec
from zvec import (
    Collection,
    CollectionOption,
    DataType,
    Doc,
    FieldSchema,
    FlatIndexParam,
    HnswIndexParam,
    HnswQueryParam,
    IVFIndexParam,
    IVFQueryParam,
    MetricType,
    OptimizeOption,
    QuantizerParam,
    Query,
    VectorSchema,
)
from zvec.typing import QuantizeType

# ==================== Constants ====================

DIM = 32
NUM_DOCS = 600
NUM_QUERIES = 20
TOPK = 10


# ==================== Helpers ====================


def make_vectors(num: int, dim: int, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal((num, dim)).astype(np.float32)


def make_docs(vectors: np.ndarray) -> list[Doc]:
    return [
        Doc(
            id=f"{i}",
            fields={"id": i},
            vectors={"embedding": vectors[i].tolist()},
        )
        for i in range(len(vectors))
    ]


def brute_force_topk_l2(
    vectors: np.ndarray, query: np.ndarray, topk: int
) -> list[str]:
    dists = np.sum((vectors - query) ** 2, axis=1)
    return [f"{i}" for i in np.argsort(dists)[:topk]]


def make_schema(name: str, index_param) -> zvec.CollectionSchema:
    return zvec.CollectionSchema(
        name=name,
        fields=[FieldSchema("id", DataType.INT64, nullable=False)],
        vectors=[
            VectorSchema(
                "embedding",
                DataType.VECTOR_FP32,
                dimension=DIM,
                index_param=index_param,
            ),
        ],
    )


def create_collection(tmp_path_factory, name: str, index_param) -> Collection:
    temp_dir = tmp_path_factory.mktemp("zvec_pq")
    coll = zvec.create_and_open(
        path=str(temp_dir / name),
        schema=make_schema(name, index_param),
        option=CollectionOption(read_only=False, enable_mmap=True),
    )
    assert coll is not None, f"Failed to create and open collection {name}"
    return coll


def average_recall(
    coll: Collection, vectors: np.ndarray, query_param
) -> float:
    """Average recall@TOPK over the first NUM_QUERIES stored vectors."""
    hits = 0
    total = 0
    for qi in range(NUM_QUERIES):
        expected = set(brute_force_topk_l2(vectors, vectors[qi], TOPK))
        result = coll.query(
            Query(
                field_name="embedding",
                vector=vectors[qi].tolist(),
                param=query_param,
            ),
            topk=TOPK,
        )
        assert len(result) == TOPK
        hits += len(expected & {doc.id for doc in result})
        total += TOPK
    return hits / total


# ==================== Fixtures ====================


@pytest.fixture(scope="module")
def dataset() -> np.ndarray:
    return make_vectors(NUM_DOCS, DIM)


@pytest.fixture(scope="module")
def docs(dataset: np.ndarray) -> list[Doc]:
    return make_docs(dataset)


# ==================== HNSW + PQ ====================


class TestHnswPqCollection:
    @pytest.mark.parametrize(
        "num_bits,min_recall",
        [(8, 0.6), (4, 0.3)],
        ids=["pq_int8", "pq_int4"],
    )
    def test_insert_query_recall(
        self, tmp_path_factory, dataset, docs, num_bits, min_recall
    ):
        coll = create_collection(
            tmp_path_factory,
            f"test_hnsw_pq_int{num_bits}",
            HnswIndexParam(
                metric_type=MetricType.L2,
                m=16,
                ef_construction=200,
                quantize_type=QuantizeType.PQ,
                quantizer_param=QuantizerParam(num_chunk=8, num_bits=num_bits),
            ),
        )
        try:
            result = coll.insert(docs)
            assert len(result) == len(docs)
            for item in result:
                assert item.ok()
            assert coll.stats.doc_count == len(docs)

            # Before optimize the segment is still served by brute force, so
            # the PQ code path is only exercised after the quantized index is
            # built.
            coll.optimize(option=OptimizeOption())

            recall = average_recall(coll, dataset, HnswQueryParam(ef=200))
            assert recall >= min_recall, (
                f"HNSW+PQ int{num_bits} recall@{TOPK} too low: {recall:.3f}"
            )
        finally:
            coll.destroy()

    def test_int8_beats_int4(self, tmp_path_factory, dataset, docs):
        """num_bits must actually reach the quantizer: 256 centroids per
        sub-space have to outperform 16."""
        recalls = {}
        for num_bits in (8, 4):
            coll = create_collection(
                tmp_path_factory,
                f"test_hnsw_pq_cmp_int{num_bits}",
                HnswIndexParam(
                    metric_type=MetricType.L2,
                    m=16,
                    ef_construction=200,
                    quantize_type=QuantizeType.PQ,
                    quantizer_param=QuantizerParam(
                        num_chunk=8, num_bits=num_bits
                    ),
                ),
            )
            try:
                for item in coll.insert(docs):
                    assert item.ok()
                coll.optimize(option=OptimizeOption())
                recalls[num_bits] = average_recall(
                    coll, dataset, HnswQueryParam(ef=200)
                )
            finally:
                coll.destroy()
        assert recalls[8] > recalls[4], (
            f"int8 recall {recalls[8]:.3f} should exceed int4 "
            f"{recalls[4]:.3f}"
        )

    def test_persistence_reopen(self, tmp_path_factory, dataset, docs):
        temp_dir = tmp_path_factory.mktemp("zvec_pq_reopen")
        path = str(temp_dir / "test_hnsw_pq_reopen")
        option = CollectionOption(read_only=False, enable_mmap=True)

        coll = zvec.create_and_open(
            path=path,
            schema=make_schema(
                "test_hnsw_pq_reopen",
                HnswIndexParam(
                    metric_type=MetricType.L2,
                    m=16,
                    ef_construction=200,
                    quantize_type=QuantizeType.PQ,
                    quantizer_param=QuantizerParam(num_chunk=8, num_bits=8),
                ),
            ),
            option=option,
        )
        result = coll.insert(docs)
        for item in result:
            assert item.ok()
        # Builds the quantized index and persists the trained PQ codebook.
        coll.optimize(option=OptimizeOption())
        coll.flush()

        recall_before = average_recall(coll, dataset, HnswQueryParam(ef=200))
        assert recall_before >= 0.6
        del coll

        # Reopen: the PQ codebook must be restored from the persisted index,
        # otherwise recall would collapse.
        reopened = zvec.open(path=path, option=option)
        try:
            assert reopened.stats.doc_count == len(docs)
            recall_after = average_recall(
                reopened, dataset, HnswQueryParam(ef=200)
            )
            assert recall_after == pytest.approx(recall_before, abs=0.05), (
                f"recall changed after reopen: {recall_after:.3f} "
                f"(before: {recall_before:.3f})"
            )
        finally:
            reopened.destroy()


# ==================== IVF + PQ ====================


class TestIvfPqCollection:
    @pytest.mark.parametrize(
        "num_bits,min_recall",
        [(8, 0.6), (4, 0.3)],
        ids=["pq_int8", "pq_int4"],
    )
    def test_insert_optimize_query_recall(
        self, tmp_path_factory, dataset, docs, num_bits, min_recall
    ):
        coll = create_collection(
            tmp_path_factory,
            f"test_ivf_pq_int{num_bits}",
            IVFIndexParam(
                metric_type=MetricType.L2,
                n_list=8,
                n_iters=5,
                quantize_type=QuantizeType.PQ,
                quantizer_param=QuantizerParam(num_chunk=8, num_bits=num_bits),
            ),
        )
        try:
            result = coll.insert(docs)
            assert len(result) == len(docs)
            for item in result:
                assert item.ok()

            # Build the IVF index; the PQ quantizer is trained on the
            # original vectors here.
            coll.optimize(option=OptimizeOption())

            # Probe all lists so recall loss comes from PQ only.
            recall = average_recall(coll, dataset, IVFQueryParam(nprobe=8))
            assert recall >= min_recall, (
                f"IVF+PQ int{num_bits} recall@{TOPK} too low: {recall:.3f}"
            )
        finally:
            coll.destroy()


# ==================== Schema validation ====================


class TestPqSchemaValidation:
    def test_flat_pq_rejected(self, tmp_path_factory):
        """PQ is only wired into HNSW / IVF; FLAT must be rejected."""
        with pytest.raises(Exception):
            create_collection(
                tmp_path_factory,
                "test_flat_pq_invalid",
                FlatIndexParam(
                    metric_type=MetricType.L2,
                    quantize_type=QuantizeType.PQ,
                    quantizer_param=QuantizerParam(num_chunk=8),
                ),
            )

    def test_invalid_num_bits_rejected(self, tmp_path_factory):
        with pytest.raises(Exception):
            create_collection(
                tmp_path_factory,
                "test_pq_bad_bits",
                HnswIndexParam(
                    metric_type=MetricType.L2,
                    quantize_type=QuantizeType.PQ,
                    quantizer_param=QuantizerParam(num_chunk=8, num_bits=6),
                ),
            )

    def test_indivisible_num_chunk_rejected(self, tmp_path_factory):
        # DIM=32 is not divisible by num_chunk=7
        with pytest.raises(Exception):
            create_collection(
                tmp_path_factory,
                "test_pq_bad_chunk",
                HnswIndexParam(
                    metric_type=MetricType.L2,
                    quantize_type=QuantizeType.PQ,
                    quantizer_param=QuantizerParam(num_chunk=7),
                ),
            )
