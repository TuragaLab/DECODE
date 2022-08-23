from decode.generic import tensor


def test_tensor_memory_mapped():
    assert hasattr(tensor.TensorMemoryMapped, "size")