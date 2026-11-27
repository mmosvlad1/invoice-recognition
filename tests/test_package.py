"""What the recognition package offers to its callers."""

from invoice_recognition import recognition
from invoice_recognition.recognition import Config, Result, config, pipeline, process, results


def test_the_flow_is_exported_from_the_package() -> None:
    assert sorted(recognition.__all__) == ["Config", "Result", "process"]
    assert process is pipeline.process
    assert Config is config.Config and Result is results.Result
