# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The pure parts of the NVFP4 time-budget harness that decide what a measurement MEANS."""

import importlib.util
import sys
import types
from pathlib import Path

import pytest

_SCRIPTS = Path(__file__).resolve().parents[3] / "scripts"


def _script(name: str):
    """One harness script as a module. Imported by path: ``scripts/`` is not a package."""
    path = _SCRIPTS / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_the_nvfp4_gemm_is_not_charged_to_attention():
    # "flash" is a substring of "flashinfer": testing attention first would file the NVFP4 GEMM under attention.
    profile = _script("nvfp4_budget_profile")
    fp4 = "flashinfer::DeviceGemmFp4_128x128"
    assert profile.classify(fp4, "phase:denoise") == "fp4_gemm"
    assert profile.classify(fp4, None) == "fp4_gemm"
    assert profile.classify("nvfp4_quantize_with_block_size", "phase:denoise") == "fp4_quantize"
    assert (
        profile.classify(
            "cudnn_generated_fort_native_sdpa_sm100_flash_fprop_f16_knob_36", "phase:denoise"
        )
        == "attention_cudnn"
    )
    assert (
        profile.classify(
            "void pytorch_flash::flash_fwd_kernel<Flash_fwd_kernel_traits<128", "phase:denoise"
        )
        == "attention_flash"
    )
    assert (
        profile.classify("fmha_cutlassF_bf16_aligned_64x128_rf_sm80", "phase:denoise")
        == "attention_mem_efficient"
    )


def test_only_the_bf16_barrier_fill_is_charged_to_the_nvfp4_barrier():
    # The fp8 arms fire no barrier at all, yet a bare "FillFunctor" test charged their ordinary
    # fills to it (z-image fp8 reported 109 barrier fills per render, the same 109 the nvfp4 arm
    # carries on top of its 306 fp4 GEMMs).
    profile = _script("nvfp4_budget_profile")
    barrier = (
        "void at::native::vectorized_elementwise_kernel<8, at::native::FillFunctor<c10::BFloat16>,"
        " std::array<char*, 1ul> >(int, at::native::FillFunctor<c10::BFloat16>,"
        " std::array<char*, 1ul>)"
    )
    generic = (
        "void at::native::vectorized_elementwise_kernel<4, at::native::FillFunctor<float>,"
        " std::array<char*, 1ul> >(int, at::native::FillFunctor<float>, std::array<char*, 1ul>)"
    )
    assert profile.classify(barrier, "phase:denoise") == "barrier_fill"
    assert profile.classify(barrier, None) == "elementwise_eager"
    assert profile.classify(generic, "phase:denoise") == "elementwise_eager"
    assert profile.classify(generic, "phase:vae") == "vae_decode"


def test_each_graph_arm_writes_its_own_latent_and_trace():
    # --graphs both runs two arms in one invocation: a shared path would leave the graphs-off
    # tensor in the file both cells name, so the cross-arm comparison would compare it with itself.
    profile = _script("nvfp4_budget_profile")
    on = profile.arm_path("/tmp/z.pt", "on", both = True)
    off = profile.arm_path("/tmp/z.pt", "off", both = True)
    assert on != off
    assert (on, off) == ("/tmp/z_graphson.pt", "/tmp/z_graphsoff.pt")
    assert profile.arm_path("/tmp/z.pt", "on", both = False) == "/tmp/z.pt"
    assert profile.arm_path("/tmp/z_{graphs}.pt", "off", both = True) == "/tmp/z_off.pt"


def test_an_inductor_kernel_is_inductor_time_whatever_it_was_fused_from():
    profile = _script("nvfp4_budget_profile")
    fused = "triton_poi_fused__scaled_dot_product_cudnn_attention_add_7"
    assert profile.classify(fused, "phase:denoise") == "inductor_triton"


def test_the_phase_window_outranks_the_kernel_name():
    # Window overrides keep a text-encoder GEMM out of the denoise fp8 bucket.
    profile = _script("nvfp4_budget_profile")
    gemm = "nvjet_tst_128x128_64x4_1x1_v_bz_coopA_NTn"
    assert profile.classify(gemm, "phase:te") == "text_encoder"
    assert profile.classify(gemm, "phase:vae") == "vae_decode"
    assert profile.classify(gemm, "phase:denoise") == "fp8_scaled_mm"
    assert profile.classify(gemm, None) == "gemm_other"


def test_gpu_busy_unions_overlapping_intervals_instead_of_summing_them(tmp_path):
    # Two 10 ms kernels overlapping by 5 ms are 15 ms of busy GPU, not 20; a sum would exceed the wall clock.
    profile = _script("nvfp4_budget_profile")
    trace = tmp_path / "trace.json"
    trace.write_text(
        '{"traceEvents": ['
        '{"ph": "X", "cat": "kernel", "ts": 0.0, "dur": 10000.0, "name": "a"},'
        '{"ph": "X", "cat": "kernel", "ts": 5000.0, "dur": 10000.0, "name": "b"},'
        '{"ph": "X", "cat": "cpu_op", "ts": 0.0, "dur": 99000.0, "name": "host"}'
        "]}"
    )
    busy = profile._union_busy_us(trace)
    assert busy["busy_us"] == pytest.approx(15000.0)
    assert busy["sum_dur_us"] == pytest.approx(20000.0)
    assert busy["n_intervals"] == 2
    assert profile._union_busy_us(trace, window = (10**9, 2 * 10**9))["busy_us"] == 0.0


def test_paired_times_reports_the_direction_it_claims():
    # speedup > 1 means the SECOND argument is faster; backwards would invert every A/B verdict.
    profile = _script("nvfp4_budget_profile")
    slow = [1.0, 1.1, 1.2]
    fast = [0.5, 0.55, 0.6]
    row = profile.paired_times(slow, fast)
    assert row["speedup_p50"] == pytest.approx(2.0)
    assert row["wins"] == 3
    assert row["n"] == 3
    assert profile.paired_times(slow, slow)["speedup_p50"] == pytest.approx(1.0)


def test_the_summariser_reads_a_results_directory_and_recomputes_nothing(tmp_path):
    summarise = _script("nvfp4_budget_summarise")
    (tmp_path / "cell_a.json").write_text(
        '{"tag": "cell_a", "arm": "nvfp4", "graphs": "on", "model": "m", "steps": 4,'
        ' "resolution": "1024", "p50_s": 0.5, "min_s": 0.49, "gpu_busy_union_s": 0.4,'
        ' "host_idle_s": 0.1, "gpu_busy_fraction_of_wall": 0.8, "clean": true,'
        ' "profiler_overhead_ratio": 1.1, "phase_sync_overhead_s": 0.01,'
        ' "contention": {"pre": {"verdict": "clean"}, "post": {"verdict": "clean"}},'
        ' "buckets": {"fp4_gemm": {"calls_per_render": 8, "ms_per_render": 12.5,'
        ' "pct_busy": 3.1}}, "attention_by_kernel": [], "memcpy_d2d":'
        ' {"calls_per_render": 2, "calls_per_step": 0.5, "ms_per_render": 0.1}}\n'
    )
    order = tmp_path / "order.txt"
    order.write_text("# the pass, in order\ncell_a\ncell_that_never_ran\n")
    notes = tmp_path / "notes.md"
    notes.write_text("## Caveats\n\nThe card was shared.\n")
    out = tmp_path / "report" / "budget.md"
    assert (
        summarise.main(
            [
                "--results-dir",
                str(tmp_path),
                "--out",
                str(out),
                "--order",
                str(order),
                "--notes",
                str(notes),
                "--title",
                "A pass",
            ]
        )
        == 0
    )
    text = out.read_text()
    assert text.startswith("# A pass")
    assert "| cell_a | nvfp4 | on | 0.5000 | 0.4000 | 0.1000 | 80.0% | True |" in text
    assert "| fp4_gemm | 8 | 12.50 | 3.1% |" in text
    assert "The card was shared." in text
    assert "cell_that_never_ran" not in text


def _cell(tag: str, n_timed: int, n_profiled: int) -> str:
    """One minimal cell JSON with the render counts the profiler actually recorded."""
    walls = ", ".join(["0.5"] * n_timed)
    profiled = ", ".join(["0.6"] * n_profiled)
    return (
        f'{{"tag": "{tag}", "arm": "fp8", "graphs": "on", "model": "m", "steps": 4,'
        f' "resolution": "1024", "p50_s": 0.5, "min_s": 0.49, "gpu_busy_union_s": 0.4,'
        f' "host_idle_s": 0.1, "gpu_busy_fraction_of_wall": 0.8, "clean": true,'
        f' "profiler_overhead_ratio": 1.1, "phase_sync_overhead_s": 0.01,'
        f' "unprofiled_s": [{walls}], "profiled_walls_s": [{profiled}],'
        f' "contention": {{"pre": {{"verdict": "clean"}}, "post": {{"verdict": "clean"}}}},'
        f' "buckets": {{}}, "attention_by_kernel": [], "memcpy_d2d":'
        f' {{"calls_per_render": 2, "calls_per_step": 0.5, "ms_per_render": 0.1}}}}\n'
    )


def test_the_report_states_each_cells_own_render_counts(tmp_path):
    # The video cells ran --timed 3 --profiled 1 next to the image cells' 7 and 2, so a fixed
    # "median of 7 unprofiled renders" preamble mislabels the protocol of half the report.
    summarise = _script("nvfp4_budget_summarise")
    (tmp_path / "image_cell.json").write_text(_cell("image_cell", 7, 2))
    (tmp_path / "video_cell.json").write_text(_cell("video_cell", 3, 1))
    order = tmp_path / "order.txt"
    order.write_text("image_cell\nvideo_cell\n")
    out = tmp_path / "budget.md"
    argv = ["--results-dir", str(tmp_path), "--out", str(out), "--order", str(order)]
    assert summarise.main(argv) == 0
    text = out.read_text()
    assert "median of 7 unprofiled renders" not in text
    assert "of 7 unprofiled renders" in text
    assert "of 3 unprofiled renders" in text
    assert "of 2 profiled" in text
    assert "of 1 profiled" in text


def test_every_harness_script_parses_its_arguments_without_a_gpu():
    # torch, diffusers and the Studio backend are imported inside main(), so --help works without them.
    for name in (
        "nvfp4_budget_profile",
        "nvfp4_budget_attention_ab",
        "nvfp4_budget_summarise",
        "nvfp4_budget_vae_numerics",
    ):
        module = _script(name)
        with pytest.raises(SystemExit) as exc:
            module.main(["--help"])
        assert exc.value.code == 0


class _Processor:
    def __init__(self, backend):
        self._attention_backend = backend


class _Attn:
    def __init__(self, backend):
        self.processor = _Processor(backend)


class _Denoiser:
    """The two levels ``observed_backends`` walks: a module tree whose attention submodules hold a processor."""

    def __init__(self, *backends):
        self._subs = [_Attn(b) for b in backends]

    def modules(self):
        return [self, *self._subs]


def test_a_refused_attention_switch_is_dropped_before_the_arm_is_timed():
    # set_attention_backend leaves the PREVIOUS backend installed when it refuses, so a render that merely succeeds
    # would publish the old kernel's time under the new kernel's label.
    ab = _script("nvfp4_budget_attention_ab")
    ok = {"requested": "_native_flash", "observed": ["_native_flash"], "errors": []}
    assert ab.switch_failure(ok) is None
    refused = {
        "requested": "_native_flash",
        "observed": ["_native_cudnn"],
        "errors": ["FluxTransformer2DModel: ValueError: `backend=` must be one of"],
    }
    assert "refused" in ab.switch_failure(refused)
    silent = {"requested": "_native_flash", "observed": ["_native_cudnn"], "errors": []}
    assert "not _native_flash" in ab.switch_failure(silent)
    assert ab.switch_failure({"requested": "_native_flash", "observed": [], "errors": []}) is None


def test_the_observed_backend_is_read_off_the_processors_not_the_denoiser():
    # diffusers writes processor._attention_backend; the denoiser module carries no such attribute.
    ab = _script("nvfp4_budget_attention_ab")
    backend = types.SimpleNamespace(value = "_native_cudnn")
    assert ab.observed_backends([_Denoiser(backend, backend)]) == ["_native_cudnn"]
    assert ab.observed_backends([_Denoiser(None)]) == []
    mixed = _Denoiser(backend, types.SimpleNamespace(value = "_native_flash"))
    assert ab.observed_backends([mixed]) == ["_native_cudnn", "_native_flash"]


def test_compile_off_loads_the_eager_tier_rather_than_relabelling_a_compiled_run():
    profile = _script("nvfp4_budget_profile")
    assert profile.resolve_load_speed_mode("off", "default") == "eager"
    assert profile.resolve_load_speed_mode("off", "max") == "eager"
    assert profile.resolve_load_speed_mode("regional", "default") == "default"
    assert profile.resolve_load_speed_mode("whole", "max") == "max"


def _both_arms_argv(tmp_path, out: str) -> list:
    return [
        "--family",
        "z-image",
        "--model",
        "m",
        "--arm",
        "fp8",
        "--steps",
        "4",
        "--graphs",
        "both",
        "--backend-root",
        str(tmp_path / "backend"),
        "--out",
        out,
    ]


def _patched_main(profile, monkeypatch, checks: list, seen: list):
    """``main`` with the two GPU-touching calls replaced, so the per-arm bookkeeping is testable."""

    def fake_check(tag, out_dir):
        checks.append((tag, out_dir))
        return {"verdict": "clean" if len(checks) == 1 else "CONTENDED", "slice_ratio": 1.0}

    def fake_run_one(
        args,
        graphs,
        pre,
        root,
        out_path = None,
    ):
        if out_path is None:
            out_path = profile.arm_path(args.out, graphs, both = args.graphs == "both")
        seen.append((graphs, pre["verdict"], Path(out_path)))
        Path(out_path).write_text("{}\n")
        return 0

    monkeypatch.setattr(profile, "contention_check", fake_check)
    monkeypatch.setattr(profile, "run_one", fake_run_one)


def test_each_graph_arm_takes_its_own_contention_pre_check(tmp_path, monkeypatch):
    # One pre-check shared by both arms backdates the second arm's window: a neighbour that
    # arrives during arm one and leaves before arm two's post-check marks arm two clean.
    profile = _script("nvfp4_budget_profile")
    checks: list = []
    seen: list = []
    _patched_main(profile, monkeypatch, checks, seen)
    out = str(tmp_path / "cell_graphs{graphs}.json")
    assert profile.main(_both_arms_argv(tmp_path, out)) == 0
    assert [tag for tag, _dir in checks] == ["pre", "pre"]
    assert [(graphs, verdict) for graphs, verdict, _p in seen] == [
        ("on", "clean"),
        ("off", "CONTENDED"),
    ]


def test_each_graph_arm_gets_its_output_directory_created(tmp_path, monkeypatch):
    # {graphs} is allowed anywhere in --out, so the parent of the FORMATTED path is the one that
    # has to exist; creating the template's parent leaves a literal {graphs} dir and the arm's
    # first checkpoint write fails after the model has already loaded.
    profile = _script("nvfp4_budget_profile")
    checks: list = []
    seen: list = []
    _patched_main(profile, monkeypatch, checks, seen)
    out = str(tmp_path / "results" / "{graphs}" / "cell.json")
    assert profile.main(_both_arms_argv(tmp_path, out)) == 0
    assert [p for _g, _v, p in seen] == [
        tmp_path / "results" / "on" / "cell.json",
        tmp_path / "results" / "off" / "cell.json",
    ]
    for _graphs, _verdict, path in seen:
        assert path.read_text() == "{}\n"
    assert not (tmp_path / "results" / "{graphs}").exists()


def _gen_args(**overrides):
    import argparse

    base = dict(
        backend = "image",
        prompt = "a red sailboat",
        resolution = "1024",
        steps = 8,
        seed = 1234,
        frames = None,
        negative_prompt = None,
    )
    base.update(overrides)
    return argparse.Namespace(**base)


def test_the_image_path_renders_with_the_negative_prompt_it_was_given():
    # The image backend's generate() takes negative_prompt; parsing --negative-prompt and then
    # dropping it renders the positive-only case under the label of the requested one.
    profile = _script("nvfp4_budget_profile")
    image = profile.generation_kwargs(_gen_args(negative_prompt = "blurry, watermark"), 4.0)
    assert image["negative_prompt"] == "blurry, watermark"
    assert image["batch_size"] == 1
    video = profile.generation_kwargs(
        _gen_args(
            backend = "video",
            resolution = "1280x704",
            frames = 121,
            negative_prompt = "blurry, watermark",
        ),
        5.0,
    )
    assert video["negative_prompt"] == "blurry, watermark"
    assert (video["width"], video["height"], video["num_frames"]) == (1280, 704, 121)
    assert "negative_prompt" not in profile.generation_kwargs(_gen_args(), 4.0)


def test_the_attention_ab_starts_a_different_backend_each_round():
    # A fixed order pins each backend to the same position in every round, so drift across the
    # round lands on the same one every time; pairing by seed does not remove that.
    ab = _script("nvfp4_budget_attention_ab")
    live = ["cudnn", "flash", "efficient"]
    orders = [ab.rotated(live, rot) for rot in range(4)]
    assert orders[0] == live
    assert len({tuple(o) for o in orders[:3]}) == 3
    assert [o[0] for o in orders] == ["cudnn", "flash", "efficient", "cudnn"]
    for order in orders:
        assert sorted(order) == sorted(live)
    assert ab.rotated([], 2) == []


def test_the_vae_numerics_eager_arm_unwraps_a_decode_compiled_at_load():
    # UNSLOTH_DIFFUSION_COMPILE_VAE=0 is ignored for a U-Net pipe, so the load can hand back a
    # compiled vae.decode; timing that as the eager arm compares compiled against compiled.
    numerics = _script("nvfp4_budget_vae_numerics")

    def decode(latent):
        return latent

    def wrapper(latent):
        return decode(latent)

    wrapper._torchdynamo_orig_callable = decode
    wrapper._torchdynamo_wrapper_id = id(wrapper)
    eager, unwrapped = numerics.eager_decode_of(wrapper)
    assert (eager, unwrapped) == (decode, True)
    assert numerics.eager_decode_of(decode) == (decode, False)

    def copied(latent):
        return decode(latent)

    # A functools.wraps copy carries the attribute without the matching id and is not a wrapper.
    copied._torchdynamo_orig_callable = decode
    copied._torchdynamo_wrapper_id = id(wrapper)
    assert numerics.eager_decode_of(copied) == (copied, False)
