"""Validation for measured learning-lab GPU results."""


def validate_execution_result(document: dict[str, object]) -> dict[str, object]:
    if document.get("provider") not in {"apple-mlx", "modal-triton"}:
        raise ValueError("execution result must identify a real GPU provider")

    required = {
        "schema_version",
        "experiment",
        "provider",
        "device",
        "implementation",
        "workload",
        "correctness",
        "measurements",
        "output",
    }
    if set(document) != required:
        raise ValueError("execution result fields do not match schema version 1.0.0")
    if document["schema_version"] != "1.0.0" or document["experiment"] != "paint-pixels":
        raise ValueError("unsupported execution result schema or experiment")

    workload = document["workload"]
    correctness = document["correctness"]
    measurements = document["measurements"]
    output = document["output"]
    if not isinstance(workload, dict) or set(workload) != {"pixels", "dtype", "group_size"}:
        raise ValueError("invalid execution workload")
    if workload["pixels"] <= 0 or workload["dtype"] != "float32" or workload["group_size"] <= 0:
        raise ValueError("invalid execution workload")
    if not isinstance(correctness, dict) or set(correctness) != {"passed", "max_abs_error"}:
        raise ValueError("invalid correctness result")
    if correctness["passed"] is not True or correctness["max_abs_error"] < 0:
        raise ValueError("GPU execution did not pass correctness")
    if not isinstance(measurements, list) or len(measurements) != 1:
        raise ValueError("execution must contain one latency measurement")
    latency = measurements[0]
    if latency.get("name") != "latency" or latency.get("unit") != "ms" or latency.get("value", -1) < 0:
        raise ValueError("invalid latency measurement")
    if not isinstance(output, dict) or set(output) != {"checksum"}:
        raise ValueError("invalid output summary")
    return document
