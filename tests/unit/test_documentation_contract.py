"""Regression tests for the Gate C documentation and scope contract."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _read(relative_path: str) -> str:
    return (ROOT / relative_path).read_text(encoding="utf-8")


def _normalized(relative_path: str) -> str:
    return " ".join(_read(relative_path).split())


def test_active_registry_matches_reframed_scope() -> None:
    registry = _read("docs/experiment_registry.md")

    for experiment_id in range(1, 9):
        assert f"| C{experiment_id:02d} |" in registry

    for removed_id in ("C09", "C10", "M01", "M08"):
        assert f"| {removed_id} |" not in registry

    assert "| C11 | candidate RQ2 mechanism |" in registry
    assert "IMPLEMENTED / EXECUTED / NOT VERIFIED / NOT REPORTED" in registry
    assert "MALT experiments M01--M08 are deferred and not implemented" in registry


def test_readme_and_protocol_use_gate_n_claim_boundary() -> None:
    readme = _normalized("README.md")
    protocol = _normalized("docs/protocol.md")

    assert "**In progress:** the MALT study" not in readme
    assert "Gate N returned `REFRAME`" in readme
    assert "C01--C04 are replication and benchmark context" in protocol
    assert "C05--C08 are exploratory or appendix evidence" in protocol
    assert "**RQ1 -- benchmark context:**" in protocol
    assert "**RQ2 -- candidate primary mechanism:**" in protocol
    assert "**RQ3 -- optional supporting diagnostic:**" in protocol
    for retired_question in range(4, 9):
        assert f"**RQ{retired_question}:**" not in protocol


def test_future_experiments_remain_blocked_and_oracle_labeled() -> None:
    protocol = _normalized("docs/protocol.md")
    results = _read("docs/RESULTS.md")

    assert "The completed C11-A review" in protocol
    assert "implementation-readiness reviews passed" in protocol
    oracle_label = "ORACLE / DIAGNOSTIC - NOT A DEPLOYMENT METHOD"
    assert oracle_label in protocol
    assert "| C11 | IMPLEMENTED | EXECUTED |" in results
    assert "| C12 | NOT IMPLEMENTED | NOT EXECUTED |" in results


def test_all_adrs_declare_status_and_malt_is_deferred() -> None:
    adr_paths = sorted((ROOT / "docs" / "decisions").glob("ADR-*.md"))

    assert adr_paths
    for adr_path in adr_paths:
        assert "## Status" in adr_path.read_text(encoding="utf-8"), adr_path.name

    malt_adr = _read("docs/decisions/ADR-0002-malt-group-split.md")
    assert "Deferred -- not implemented" in malt_adr


def test_runbook_covers_every_completed_experiment() -> None:
    runbook = _read("docs/RUNBOOK.md")

    expected = {
        "c01": ("run_c01_baseline.py", "fd001_baseline.yaml"),
        "c02": ("run_c02_conformal.py", "conformal.yaml"),
        "c03": ("run_c03_cqr.py", "cqr.yaml"),
        "c04": ("run_c04_shift_matrix.py", "shift_matrix.yaml"),
        "c05": ("run_c05_weighted_conformal.py", "weighted_conformal.yaml"),
        "c06": ("run_c06_sensitivity.py", "sensitivity.yaml"),
        "c07": ("run_c07_regime_scaling.py", "regime_scaling.yaml"),
        "c08": ("run_c08_adaptive_conformal.py", "adaptive_conformal.yaml"),
    }
    for script_name, config_name in expected.values():
        assert f"scripts/{script_name}" in runbook
        assert f"configs/cmapss/{config_name}" in runbook


def test_reporting_status_cannot_be_inferred_from_files() -> None:
    reproducibility = _read("REPRODUCIBILITY.md")
    results = _normalized("docs/RESULTS.md")

    for status in ("PLANNED", "IMPLEMENTED", "EXECUTED", "VERIFIED", "REPORTED"):
        assert f"`{status}`" in reproducibility

    assert 'float_precision="round_trip"' in results
    assert "C01--C08 are `IMPLEMENTED`, `EXECUTED`, `VERIFIED`, and `REPORTED`" in reproducibility
    assert "reports/results/summary.csv" in results
    assert "they are never converted to zero" in results
    assert "the aggregate remains `PENDING`" in results
    assert "| C11 | IMPLEMENTED | EXECUTED | NOT VERIFIED | NOT ELIGIBLE |" in results


def test_gate_d_builder_and_artifact_index_are_the_only_reporting_path() -> None:
    readme = _normalized("README.md")
    runbook = _read("docs/RUNBOOK.md")
    index = _read("docs/artifact_index.yaml")

    assert "deterministically reported" in readme
    assert "scripts/build_results.py" in runbook
    assert "docs/artifact_index.yaml" in runbook
    for experiment_id in range(1, 9):
        assert f"experiment_id: C{experiment_id:02d}" in index
    assert index.count("status: OFFICIAL") == 8
    assert "official_run_contracts:" in index
    assert "manifest_set_sha256:" in index


def test_c11_reporting_is_separate_and_not_published_before_review() -> None:
    contract = _normalized("docs/C11_REPORTING.md")
    index = _read("docs/c11_artifact_index.yaml")
    results = _read("docs/RESULTS.md")

    assert "separate from the historical C01--C08 Gate D builder" in contract
    assert "NOT YET PUBLISHED / NOT YET REPORTED" in contract
    assert "not an independent statistical reconstruction engine" in contract
    assert "must recompute" in contract
    assert "manifest_sha256:" in index
    assert "status: VERIFIED_CANDIDATE" in index
    assert "scripts/build_c11_results.py" not in _read("docs/RUNBOOK.md")
    assert "| C11 | IMPLEMENTED | EXECUTED | NOT VERIFIED | NOT ELIGIBLE |" in results


def test_gate_d_reconstruction_and_trust_anchor_are_explicit() -> None:
    runbook = _normalized("docs/RUNBOOK.md")
    results = _normalized("docs/RESULTS.md")
    reproducibility = _normalized("REPRODUCIBILITY.md")

    assert "builder_git_sha" in runbook
    assert "manifest-set digest mismatch" in runbook
    assert "checksums.sha256" in runbook
    assert "later clean revision" in results
    assert "exact official run contracts" in reproducibility


def test_gate_d_environment_is_exact_and_revalidated() -> None:
    readme = _read("README.md")
    runbook = _normalized("docs/RUNBOOK.md")
    reproducibility = _normalized("REPRODUCIBILITY.md")
    python_version = _read(".python-version").strip()
    project = _read("pyproject.toml")

    assert python_version == "3.11.9"
    assert 'requires-python = ">=3.11,<3.12"' in project
    assert "--python 3.11.9" in readme
    assert "exact `environment`" in runbook
    assert "revalidated immediately before final publication" in runbook
    assert "direct-package versions" in reproducibility


def test_c11_accepted_design_execution_is_recorded_and_verification_remains_blocked() -> None:
    adr = _normalized("docs/decisions/ADR-0012-c11-finite-reservoir-design.md")
    readme = _normalized("README.md")
    protocol = _normalized("docs/protocol.md")
    registry = _normalized("docs/experiment_registry.md")
    runbook = _read("docs/RUNBOOK.md")

    assert "Accepted design" in adr
    assert "FD001-to-FD001 mechanism audit" in adr
    assert "the 60 FD001 base-training engines" in adr
    assert "the remaining 40 FD001 training engines" in adr
    assert "The primary analysis does not simulate this variable" in adr
    assert "It integrates it exactly" in adr
    assert "P(K=k) = C(k-1, r-1) * C(N-k, n-r) / C(N, n)" in adr
    assert "Beta(r, n_cal + 1 - r)" in adr
    assert "Beta-binomial projection" in adr
    assert "not_evaluated_due_to_unattainable_finite_rank" in adr
    assert "not_evaluated_due_to_degenerate_full_reservoir_subset" in adr
    assert "random.Random(13)" in adr
    assert "rng.randint(lower, upper)" in adr
    assert "same five discrepancies separately against both references" in adr
    assert "evaluating both the left limit and right-continuous CDF" in adr
    assert "atol=1e-12, rtol=1e-12" in adr
    assert "empirical 1-Wasserstein distance, in observed cycle units" in adr
    assert "Immediately before atomic publication" in adr
    assert "unchanged clean Git revision" in adr
    assert "makes no binary `adequately_explained`" in adr
    assert "retrospectively motivated, prospectively frozen descriptive audit" in adr
    assert "does not authorize execution" in adr
    assert "implementation-readiness review passed" in readme
    assert "not confirmatory preregistration" in readme
    assert "The completed C11-A review" in protocol
    assert "No binary adequacy or material-departure classification" in protocol
    assert "| C11 | candidate RQ2 mechanism |" in registry
    assert "docs/C11_EXECUTION.md" in runbook
    execution = _normalized("docs/C11_EXECUTION.md")
    assert "cba16d0477ab19bf07ec02dd4be151e6b5fb670e" in execution
    assert "EXECUTED / NOT VERIFIED / NOT REPORTED" in execution
    assert (ROOT / "configs" / "cmapss" / "finite_reservoir.yaml").is_file()
    assert (ROOT / "src" / "calibrated_reliability" / "experiments" / "c11.py").is_file()
    assert (ROOT / "scripts" / "run_c11_finite_reservoir.py").is_file()
    assert not (ROOT / "outputs" / "c11").exists()
