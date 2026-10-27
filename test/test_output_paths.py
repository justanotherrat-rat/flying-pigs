import importlib.util
from pathlib import Path


PIPELINE_PATH = Path(__file__).resolve().parents[1] / "simulation" / "run_pipeline.py"
PIPELINE_SPEC = importlib.util.spec_from_file_location("run_pipeline_under_test", PIPELINE_PATH)
assert PIPELINE_SPEC is not None and PIPELINE_SPEC.loader is not None
run_pipeline = importlib.util.module_from_spec(PIPELINE_SPEC)
PIPELINE_SPEC.loader.exec_module(run_pipeline)


def test_relative_ns3_output_prefix_is_saved_under_sim_out(monkeypatch, tmp_path):
    ns3_root = tmp_path / "ns-3"
    ns3_root.mkdir()
    (ns3_root / "ns3").touch()
    sim_out = tmp_path / "sim-out"
    captured = {}

    monkeypatch.setattr(run_pipeline, "SIM_OUT", sim_out)
    monkeypatch.setattr(
        run_pipeline.subprocess,
        "run",
        lambda args, cwd, check: captured.update(args=args, cwd=cwd, check=check),
    )

    run_pipeline.run_ns3(ns3_root, "uav_mesh_eval", tmp_path / "exports", "sample", [])

    output_prefix = sim_out / "ns3" / "sample"
    assert f"--out={output_prefix}" in captured["args"][2]
    assert output_prefix.parent.is_dir()
