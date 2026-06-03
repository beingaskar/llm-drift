import pytest
from llm_drift.assertions import AssertionParseError, AssertionResult, AssertionRunner

runner = AssertionRunner()


def test_is_valid_json_passes_on_json():
    results = runner.run('{"key": 1}', ["output.is_valid_json()"])
    assert results[0].passed is True


def test_is_valid_json_fails_on_plain_text():
    results = runner.run("hello world", ["output.is_valid_json()"])
    assert results[0].passed is False
    assert "not valid JSON" in results[0].error


def test_contains_passes_when_substring_present():
    results = runner.run("invoice total $99", ["output.contains(\"invoice\")"])
    assert results[0].passed is True


def test_contains_fails_when_substring_absent():
    results = runner.run("nothing here", ["output.contains(\"invoice\")"])
    assert results[0].passed is False


def test_not_contains_passes_when_absent():
    results = runner.run("clean output", ["output.not_contains(\"error\")"])
    assert results[0].passed is True


def test_not_contains_fails_when_present():
    results = runner.run("an error occurred", ["output.not_contains(\"error\")"])
    assert results[0].passed is False


def test_min_length_passes_above_threshold():
    results = runner.run("a" * 50, ["output.min_length(10)"])
    assert results[0].passed is True


def test_min_length_fails_below_threshold():
    results = runner.run("hi", ["output.min_length(100)"])
    assert results[0].passed is False
    assert "2" in results[0].error   # actual length in message
    assert "100" in results[0].error


def test_max_length_passes_below_threshold():
    results = runner.run("short", ["output.max_length(100)"])
    assert results[0].passed is True


def test_max_length_fails_above_threshold():
    results = runner.run("a" * 200, ["output.max_length(50)"])
    assert results[0].passed is False


def test_unknown_assertion_raises_parse_error():
    with pytest.raises(AssertionParseError) as exc_info:
        runner.run("output", ["output.does_not_exist()"])
    assert "does_not_exist" in str(exc_info.value)


def test_malformed_assertion_raises_parse_error():
    with pytest.raises(AssertionParseError):
        runner.run("output", ["not_a_valid_expression"])


def test_multiple_assertions_all_evaluated():
    results = runner.run('{"a": 1}', [
        "output.is_valid_json()",
        "output.contains(\"a\")",
        "output.min_length(5)",
    ])
    assert len(results) == 3
    assert all(r.passed for r in results)


def test_returns_assertion_result_instances():
    results = runner.run("hello", ["output.contains(\"hello\")"])
    assert isinstance(results[0], AssertionResult)
