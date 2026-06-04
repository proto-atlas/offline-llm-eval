from offline_llm_eval.run.usage import UsageSource, UsageSummary, resolve_usage_summary


def test_usageがない場合はunavailableになる() -> None:
    summary = resolve_usage_summary(None)

    assert summary == UsageSummary(
        usage_source=UsageSource.UNAVAILABLE,
        usage_json=None,
    )


def test_provider_usageがある場合はprovider_reportedになる() -> None:
    summary = resolve_usage_summary({"input_tokens": 10, "output_tokens": 5})

    assert summary == UsageSummary(
        usage_source=UsageSource.PROVIDER_REPORTED,
        usage_json={"input_tokens": 10, "output_tokens": 5},
    )


def test_provider_usageは浅くcopyされる() -> None:
    usage_json = {"input_tokens": 10}
    summary = resolve_usage_summary(usage_json)

    usage_json["input_tokens"] = 20

    assert summary.usage_json == {"input_tokens": 10}


def test_usage_sourceは2値だけを公開する() -> None:
    assert {usage_source.value for usage_source in UsageSource} == {
        "unavailable",
        "provider_reported",
    }
