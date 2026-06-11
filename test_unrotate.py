"""Test inverse rotation: verify that vectors can be recovered after
rotation + quantization via the revert path."""
import os
import shutil
import tempfile
import numpy as np
import zvec
from zvec import (
    Collection,
    CollectionOption,
    DataType,
    Doc,
    FieldSchema,
    FlatIndexParam,
    MetricType,
    VectorSchema,
    OptimizeOption,
)
from zvec.typing import QuantizeType


def test_inverse_rotation_int8():
    """Test inverse rotation with INT8 streaming quantizer + rotation."""
    dim = 128
    n_docs = 10

    # Create temp dir for the collection
    tmpdir = tempfile.mkdtemp(prefix="zvec_test_inv_rotate_")
    coll_path = os.path.join(tmpdir, "collection")
    try:
        schema = zvec.CollectionSchema(
            name="test_inv_rotate",
            fields=[FieldSchema("id", DataType.INT64, nullable=False)],
            vectors=[
                VectorSchema(
                    "embedding",
                    DataType.VECTOR_FP32,
                    dimension=dim,
                    index_param=FlatIndexParam(
                        metric_type=MetricType.IP,
                        quantize_type=QuantizeType.INT8,
                        enable_rotate=True,
                    ),
                ),
            ],
        )

        collection = zvec.create_and_open(
            path=coll_path,
            schema=schema,
            option=CollectionOption(read_only=False),
        )

        # Generate random vectors
        np.random.seed(42)
        docs = []
        original_vecs = {}
        for i in range(n_docs):
            vec = np.random.randn(dim).astype(np.float32)
            vec = vec / np.linalg.norm(vec)  # Normalize for IP
            original_vecs[str(i)] = vec
            docs.append(
                Doc(
                    id=str(i),
                    fields={"id": i},
                    vectors={"embedding": vec.tolist()},
                )
            )

        # Insert
        for doc in docs:
            result = collection.insert(doc)
            assert result.ok(), f"Insert failed: {result.code()}"

        collection.flush()

        # Optimize to trigger reformer build
        collection.optimize(option=OptimizeOption())
        import time
        time.sleep(2)  # Wait for optimization to complete

        # Fetch vectors back (triggers revert path)
        ids = [str(i) for i in range(n_docs)]
        fetched = collection.fetch(ids=ids)

        assert len(fetched) == n_docs, f"Expected {n_docs} docs, got {len(fetched)}"

        # Compare fetched vectors with originals
        max_error = 0.0
        avg_error = 0.0
        print("\nDiagnostic: first fetched vector vs original:")
        for i, doc_id in enumerate(ids):
            assert doc_id in fetched, f"Doc {doc_id} not found in fetched results"
            fetched_vec = np.array(fetched[doc_id].vector("embedding"), dtype=np.float32)
            original_vec = original_vecs[doc_id]
            if i == 0:
                print(f"  fetched[:8]  = {fetched_vec[:8]}")
                print(f"  original[:8] = {original_vec[:8]}")
                print(f"  fetched shape: {fetched_vec.shape}, original shape: {original_vec.shape}")
            error = np.max(np.abs(fetched_vec - original_vec))
            avg_error += np.mean(np.abs(fetched_vec - original_vec))
            max_error = max(max_error, error)

        avg_error /= n_docs

        print(f"\n=== INT8 + Rotate Inverse Rotation Test ===")
        print(f"Max absolute error: {max_error:.6f}")
        print(f"Avg absolute error: {avg_error:.6f}")
        print(f"Number of docs: {n_docs}")

        # The error should be bounded (INT8 quantization introduces some loss)
        # With rotation, the error should still be reasonable
        assert max_error < 0.5, f"Max error {max_error} too large!"
        print("PASSED!")

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_inverse_rotation_cosine():
    """Test inverse rotation with COSINE metric + INT8 quantizer + rotation."""
    dim = 128
    n_docs = 10

    tmpdir = tempfile.mkdtemp(prefix="zvec_test_inv_rotate_cosine_")
    coll_path = os.path.join(tmpdir, "collection")
    try:
        schema = zvec.CollectionSchema(
            name="test_inv_rotate_cosine",
            fields=[FieldSchema("id", DataType.INT64, nullable=False)],
            vectors=[
                VectorSchema(
                    "embedding",
                    DataType.VECTOR_FP32,
                    dimension=dim,
                    index_param=FlatIndexParam(
                        metric_type=MetricType.COSINE,
                        quantize_type=QuantizeType.INT8,
                        enable_rotate=True,
                    ),
                ),
            ],
        )

        collection = zvec.create_and_open(
            path=coll_path,
            schema=schema,
            option=CollectionOption(read_only=False),
        )

        # Generate random vectors
        np.random.seed(42)
        docs = []
        original_vecs = {}
        for i in range(n_docs):
            vec = np.random.randn(dim).astype(np.float32)
            original_vecs[str(i)] = vec
            docs.append(
                Doc(
                    id=str(i),
                    fields={"id": i},
                    vectors={"embedding": vec.tolist()},
                )
            )

        for doc in docs:
            result = collection.insert(doc)
            assert result.ok(), f"Insert failed: {result.code()}"

        collection.flush()
        collection.optimize(option=OptimizeOption())
        import time
        time.sleep(2)

        ids = [str(i) for i in range(n_docs)]
        fetched = collection.fetch(ids=ids)

        assert len(fetched) == n_docs, f"Expected {n_docs} docs, got {len(fetched)}"

        max_error = 0.0
        avg_error = 0.0
        for doc_id in ids:
            assert doc_id in fetched, f"Doc {doc_id} not found"
            fetched_vec = np.array(fetched[doc_id].vector("embedding"), dtype=np.float32)
            original_vec = original_vecs[doc_id]
            # Normalize both for comparison (COSINE metric normalizes)
            fetched_norm = fetched_vec / (np.linalg.norm(fetched_vec) + 1e-8)
            original_norm = original_vec / (np.linalg.norm(original_vec) + 1e-8)
            error = np.max(np.abs(fetched_norm - original_norm))
            avg_error += np.mean(np.abs(fetched_norm - original_norm))
            max_error = max(max_error, error)

        avg_error /= n_docs

        print(f"\n=== COSINE + INT8 + Rotate Inverse Rotation Test ===")
        print(f"Max absolute error (normalized): {max_error:.6f}")
        print(f"Avg absolute error (normalized): {avg_error:.6f}")

        assert max_error < 0.5, f"Max error {max_error} too large!"
        print("PASSED!")

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == "__main__":
    test_inverse_rotation_int8()
    test_inverse_rotation_cosine()
    print("\n=== All tests passed! ===")
