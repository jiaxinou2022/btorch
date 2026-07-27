"""Isolated GeNN and Brian2CUDA providers for the RSNN comparison.

The frameworks in this module generate and compile standalone CUDA code. They
are deliberately executed in child processes: importing them into the main
PyTorch benchmark would mix CUDA runtime ownership and make failure recovery
unreliable. Public entry points keep preparation and result transfer outside
the reported steady-state interval.
"""

from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import torch

from benchmark.benchmark_persistent_snn import BenchCase, RSNNResult
from btorch.sparse import CSR


ExternalProvider = Literal["genn", "brian2cuda"]
MAX_ERROR_LOG_CHARS = 16_000
DEFAULT_PROVIDER_OPTIONS: dict[ExternalProvider, dict[str, object]] = {
    "genn": {
        "optimize_code": True,
    },
    "brian2cuda": {
        "sm_multiplier": 1,
        "parallel_blocks": 1,
        "calc_occupancy": True,
        "extra_threshold_kernel": True,
        "launch_bounds": False,
        "syn_launch_bounds": False,
    },
}


@dataclass(frozen=True)
class ExternalResult:
    """Results and reproducibility metadata returned by an external backend."""

    result: RSNNResult | None
    latency_samples_ms: list[float]
    framework_version: str
    build_time_s: float
    timing_method: str


def _summarize_log(output: str, max_chars: int = MAX_ERROR_LOG_CHARS) -> str:
    """Bound compiler diagnostics while retaining their beginning and end."""

    output = output.strip()
    if len(output) <= max_chars:
        return output
    head_chars = max_chars // 4
    tail_chars = max_chars - head_chars
    omitted = len(output) - max_chars
    return (
        output[:head_chars]
        + f"\n... {omitted} log characters omitted ...\n"
        + output[-tail_chars:]
    )


def _source_indices(indptr: np.ndarray) -> np.ndarray:
    """Expand source-oriented CSR row pointers into presynaptic indices."""

    return np.repeat(
        np.arange(indptr.size - 1, dtype=np.int64),
        np.diff(indptr).astype(np.int64, copy=False),
    )


def run_external_provider(
    provider: ExternalProvider,
    x_seq: torch.Tensor,
    matrix: CSR,
    case: BenchCase,
    *,
    warmup: int,
    repeat: int,
    check_correctness: bool,
    build_root: Path | None,
    timeout_s: float = 1800.0,
    provider_options: dict[str, object] | None = None,
) -> ExternalResult:
    """Run one code-generating framework in an isolated process."""

    if provider not in ("genn", "brian2cuda"):
        raise ValueError(f"Unsupported external provider: {provider}")

    retained = build_root is not None
    preserve_work = retained
    if build_root is not None:
        build_root.mkdir(parents=True, exist_ok=True)
    work = Path(
        tempfile.mkdtemp(
            prefix=f"btorch_{provider}_",
            dir=str(build_root) if build_root is not None else None,
        )
    )
    try:
        values = matrix.effective_values()
        torch.cuda.synchronize(x_seq.device)
        np.savez(
            work / "case.npz",
            x=x_seq.detach().cpu().numpy(),
            indptr=matrix.indptr.detach().cpu().numpy(),
            indices=matrix.indices.detach().cpu().numpy(),
            values=values.detach().cpu().numpy(),
        )
        options = {
            **DEFAULT_PROVIDER_OPTIONS[provider],
            **(provider_options or {}),
        }
        request = {
            "provider": provider,
            "case": asdict(case),
            "warmup": warmup,
            "repeat": repeat,
            "check_correctness": check_correctness,
            "provider_options": options,
            "work": str(work),
        }
        (work / "request.json").write_text(json.dumps(request))
        command = [
            sys.executable,
            "-m",
            "benchmark.external_rsnn_simulators",
            str(work / "request.json"),
        ]
        environment = os.environ.copy()
        if provider == "genn" and "CUDA_PATH" not in environment:
            nvcc = shutil.which("nvcc")
            if nvcc is not None:
                environment["CUDA_PATH"] = str(
                    Path(nvcc).resolve().parents[1]
                )
        stdout_path = work / "worker.stdout.log"
        stderr_path = work / "worker.stderr.log"
        status_path = work / "worker_status.jsonl"
        start = time.monotonic()
        last_status = ""
        with (
            stdout_path.open("w") as stdout_file,
            stderr_path.open("w") as stderr_file,
        ):
            process = subprocess.Popen(
                command,
                cwd=Path(__file__).resolve().parents[1],
                env=environment,
                stdout=stdout_file,
                stderr=stderr_file,
                text=True,
                start_new_session=True,
            )
            while process.poll() is None:
                if status_path.exists():
                    lines = status_path.read_text().splitlines()
                    if lines:
                        status = json.loads(lines[-1])
                        phase = str(status["phase"])
                        if phase != last_status:
                            print(
                                f"[external] provider={provider} "
                                f"phase={phase}",
                                flush=True,
                            )
                            last_status = phase
                if time.monotonic() - start > timeout_s:
                    os.killpg(process.pid, signal.SIGTERM)
                    try:
                        process.wait(timeout=5.0)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait()
                    preserve_work = True
                    raise TimeoutError(
                        f"{provider} worker exceeded {timeout_s:g}s; "
                        f"last phase={last_status or 'worker_starting'}; "
                        f"logs retained at {work}"
                    )
                time.sleep(0.25)
        stdout = stdout_path.read_text()
        stderr = stderr_path.read_text()
        if process.returncode != 0:
            detail = "\n".join(
                output
                for output in (
                    _summarize_log(stdout),
                    _summarize_log(stderr),
                )
                if output
            )
            raise RuntimeError(
                f"{provider} worker failed (exit {process.returncode}): {detail}"
            )

        metadata = json.loads((work / "result.json").read_text())
        result = None
        result_path = work / "correctness.npz"
        if result_path.exists():
            arrays = np.load(result_path)
            result = RSNNResult(
                spikes=torch.from_numpy(arrays["spikes"]).to(x_seq.device),
                v=torch.from_numpy(arrays["v"]).to(x_seq.device),
                psc=torch.from_numpy(arrays["psc"]).to(x_seq.device),
            )
        return ExternalResult(
            result=result,
            latency_samples_ms=[
                float(sample) for sample in metadata["latency_samples_ms"]
            ],
            framework_version=str(metadata["framework_version"]),
            build_time_s=float(metadata["build_time_s"]),
            timing_method=str(metadata["timing_method"]),
        )
    finally:
        if not preserve_work:
            shutil.rmtree(work, ignore_errors=True)


def _record_phase(work: Path, phase: str) -> None:
    """Append a flushed worker phase record for parent-side diagnostics."""

    record = {
        "time": time.time(),
        "phase": phase,
        "pid": os.getpid(),
    }
    with (work / "worker_status.jsonl").open("a") as file:
        file.write(json.dumps(record) + "\n")


def _patch_brian_timer(project: Path) -> None:
    """Make Brian2CUDA's per-run timer wait for all queued GPU work."""

    path = project / "network.cu"
    source = path.read_text()
    start_marker = "    start = std::chrono::high_resolution_clock::now();"
    end_marker = (
        "    Network::_globally_running = false;\n\n"
        "    current = std::chrono::high_resolution_clock::now();"
    )
    if source.count(start_marker) != 1 or source.count(end_marker) != 1:
        raise RuntimeError(
            "Unsupported Brian2CUDA network.cu timer template; expected "
            "one run timer."
        )
    source = source.replace(
        start_marker,
        "    CUDA_SAFE_CALL(cudaDeviceSynchronize());\n" + start_marker,
    )
    source = source.replace(
        end_marker,
        "    Network::_globally_running = false;\n\n"
        "    CUDA_SAFE_CALL(cudaDeviceSynchronize());\n"
        "    current = std::chrono::high_resolution_clock::now();",
    )
    path.write_text(source)


def _compile_brian_project(project: Path) -> None:
    """Compile a generated Brian2CUDA project with its release makefile."""

    process = subprocess.run(
        ["make", "-j"],
        cwd=project,
        capture_output=True,
        text=True,
        check=False,
    )
    if process.returncode != 0:
        detail = process.stderr.strip() or process.stdout.strip()
        raise RuntimeError(f"Brian2CUDA compilation failed: {detail}")


def _make_brian_network(
    arrays: dict[str, np.ndarray],
    case: BenchCase,
    project: Path,
    *,
    record: bool,
    options: dict[str, object],
):
    """Generate and compile one single-precision Brian2CUDA network."""

    import brian2 as b2
    import brian2cuda  # noqa: F401

    b2.start_scope()
    b2.device.reinit()
    b2.set_device("cuda_standalone", build_on_run=False)
    b2.prefs.core.default_float_dtype = np.float32
    b2.prefs.devices.cuda_standalone.calc_occupancy = bool(
        options["calc_occupancy"]
    )
    b2.prefs.devices.cuda_standalone.use_atomics = True
    b2.prefs.devices.cuda_standalone.SM_multiplier = int(
        options["sm_multiplier"]
    )
    parallel_blocks = int(options["parallel_blocks"])
    b2.prefs.devices.cuda_standalone.parallel_blocks = (
        None if parallel_blocks == 0 else parallel_blocks
    )
    b2.prefs.devices.cuda_standalone.extra_threshold_kernel = bool(
        options["extra_threshold_kernel"]
    )
    b2.prefs.devices.cuda_standalone.launch_bounds = bool(
        options["launch_bounds"]
    )
    b2.prefs.devices.cuda_standalone.syn_launch_bounds = bool(
        options["syn_launch_bounds"]
    )
    b2.defaultclock.dt = case.dt * b2.ms

    flat_input = np.asarray(arrays["x"], dtype=np.float32).reshape(
        case.t_steps,
        case.batch_size * case.n_neuron,
    )
    stimulus = b2.TimedArray(flat_input, dt=case.dt * b2.ms)
    neurons = b2.NeuronGroup(
        case.batch_size * case.n_neuron,
        model="v : 1\npsc : 1",
        threshold=f"v >= {case.v_threshold!r}",
        reset=(
            f"v = {case.v_reset!r}"
            if case.hard_reset
            else f"v -= {case.v_threshold - case.v_reset!r}"
        ),
        name="neurons",
    )
    decay = np.float32(np.exp(-case.dt / case.tau_syn))
    neurons.run_regularly(
        (
            f"v += {case.dt!r} * (-(v - {case.v_reset!r}) / "
            f"{case.tau_mem!r} + (psc + stimulus(t, i)) / {case.c_m!r})\n"
            f"psc *= {float(decay)!r}"
        ),
        when="groups",
        name="lif_psc_update",
    )

    source = _source_indices(arrays["indptr"])
    target = np.asarray(arrays["indices"], dtype=np.int64)
    value = np.asarray(arrays["values"], dtype=np.float32)
    if case.batch_size > 1:
        offsets = (
            np.arange(case.batch_size, dtype=np.int64) * case.n_neuron
        )[:, None]
        source = (source[None, :] + offsets).reshape(-1)
        target = (target[None, :] + offsets).reshape(-1)
        value = np.tile(value, case.batch_size)
    synapses = b2.Synapses(
        neurons,
        neurons,
        model="w : 1",
        on_pre="psc_post += w",
        name="recurrent",
    )
    synapses.connect(i=source, j=target)
    synapses.w = value

    objects = [neurons, synapses]
    monitor = None
    if record:
        monitor = b2.SpikeMonitor(neurons, name="spikes")
        objects.append(monitor)
    network = b2.Network(objects)
    network.run(
        case.t_steps * case.dt * b2.ms,
        profile=False,
        namespace={"stimulus": stimulus},
    )
    _record_phase(project.parent, f"{project.name}_code_generation")
    b2.device.build(
        directory=str(project),
        compile=False,
        run=False,
        with_output=False,
    )
    _patch_brian_timer(project)
    _record_phase(project.parent, f"{project.name}_compilation")
    _compile_brian_project(project)
    _record_phase(project.parent, f"{project.name}_build_complete")
    return b2, neurons, monitor


def _run_brian_project(
    b2,
    project: Path,
    *,
    warmup: int,
    repeat: int,
) -> list[float]:
    """Run a compiled project repeatedly and read synchronized run times."""

    work = project.parent
    for index in range(warmup):
        _record_phase(
            work,
            f"brian2cuda_warmup_{index + 1}_of_{warmup}",
        )
        b2.device.run(directory=str(project), with_output=False)
    samples = []
    for index in range(repeat):
        _record_phase(
            work,
            f"brian2cuda_timing_{index + 1}_of_{repeat}",
        )
        b2.device.run(directory=str(project), with_output=False)
        samples.append(float(b2.device._last_run_time) * 1e3)
    return samples


def _brian_correctness(
    arrays: dict[str, np.ndarray],
    case: BenchCase,
    project: Path,
    options: dict[str, object],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build a recorded Brian2CUDA run and return its trace and final state."""

    b2, neurons, monitor = _make_brian_network(
        arrays,
        case,
        project,
        record=True,
        options=options,
    )
    assert monitor is not None
    _record_phase(project.parent, "brian2cuda_correctness_run")
    b2.device.run(directory=str(project), with_output=False)
    spikes = np.zeros(
        (case.t_steps, case.batch_size, case.n_neuron),
        dtype=np.float32,
    )
    spike_t = np.rint(np.asarray(monitor.t / b2.ms) / case.dt).astype(np.int64)
    spike_i = np.asarray(monitor.i, dtype=np.int64)
    valid = (spike_t >= 0) & (spike_t < case.t_steps)
    batch = spike_i[valid] // case.n_neuron
    neuron = spike_i[valid] % case.n_neuron
    spikes[spike_t[valid], batch, neuron] = 1.0
    v = np.asarray(neurons.v[:], dtype=np.float32).reshape(
        case.batch_size,
        case.n_neuron,
    )
    psc = np.asarray(neurons.psc[:], dtype=np.float32).reshape(
        case.batch_size,
        case.n_neuron,
    )
    return spikes, v, psc


def _run_brian2cuda(request: dict, arrays: dict[str, np.ndarray]) -> dict:
    """Execute the Brian2CUDA benchmark worker."""

    import brian2
    import brian2cuda

    case = BenchCase(**request["case"])
    work = Path(request["work"])
    options = request["provider_options"]
    _record_phase(work, "brian2cuda_performance_build")
    build_start = time.perf_counter()
    b2, _, _ = _make_brian_network(
        arrays,
        case,
        work / "brian2cuda_performance",
        record=False,
        options=options,
    )
    build_time = time.perf_counter() - build_start
    _record_phase(work, "brian2cuda_warmup_and_timing")
    samples = _run_brian_project(
        b2,
        work / "brian2cuda_performance",
        warmup=request["warmup"],
        repeat=request["repeat"],
    )
    if request["check_correctness"]:
        _record_phase(work, "brian2cuda_correctness_build")
        build_start = time.perf_counter()
        result = _brian_correctness(
            arrays,
            case,
            work / "brian2cuda_correctness",
            options,
        )
        build_time += time.perf_counter() - build_start
        np.savez(work / "correctness.npz", spikes=result[0], v=result[1], psc=result[2])
    return {
        "latency_samples_ms": samples,
        "framework_version": (
            f"Brian2CUDA {getattr(brian2cuda, '__version__', 'unknown')}; "
            f"Brian2 {brian2.__version__}"
        ),
        "build_time_s": build_time,
        "timing_method": (
            "standalone_wall_cuda_device_synchronize"
            f";sm_multiplier={options['sm_multiplier']}"
            f";parallel_blocks={options['parallel_blocks']}"
            f";calc_occupancy={options['calc_occupancy']}"
            f";extra_threshold_kernel={options['extra_threshold_kernel']}"
            f";launch_bounds={options['launch_bounds']}"
            f";syn_launch_bounds={options['syn_launch_bounds']}"
        ),
    }


def _create_genn_model(
    arrays: dict[str, np.ndarray],
    case: BenchCase,
    project: Path,
    *,
    record: bool,
    options: dict[str, object],
):
    """Build and load a batched sparse PyGeNN model."""

    from pygenn import (
        GeNNModel,
        ParallelismHint,
        create_neuron_model,
        init_postsynaptic,
        init_weight_update,
    )

    project.mkdir(parents=True, exist_ok=True)
    input_size = case.t_steps * case.batch_size * case.n_neuron
    neuron_model = create_neuron_model(
        "BtorchRSNNLIF",
        params=[
            "tau_mem",
            "c_m",
            "v_reset",
            "v_threshold",
            "reset_delta",
            "psc_decay",
        ],
        vars=[("V", "scalar"), ("PSC", "scalar")],
        extra_global_params=[("input", "scalar*")],
        sim_code=(
            "const unsigned int input_idx = "
            f"((((unsigned int)rint(t / dt) % {case.t_steps}) * "
            f"{case.batch_size} + "
            f"batch) * {case.n_neuron}) + id;\n"
            "PSC = (PSC * psc_decay) + Isyn;\n"
            "const scalar current = PSC + input[input_idx];\n"
            "V += dt * (-(V - v_reset) / tau_mem + current / c_m);"
        ),
        threshold_condition_code="V >= v_threshold",
        reset_code=(
            "V = v_reset;"
            if case.hard_reset
            else "V -= reset_delta;"
        ),
    )
    # Require the CUDA backend explicitly. PyGeNN CPU-only distributions
    # otherwise silently select SingleThreadedCPU for batch size one, which
    # makes whole-connectome runs look like a hung GPU subprocess.
    model = GeNNModel(
        "float",
        f"btorch_rsnn_{'record' if record else 'perf'}",
        backend="cuda",
        optimize_code=bool(options["optimize_code"]),
    )
    model.dt = case.dt
    model.batch_size = case.batch_size
    # Use GeNN's own CUDA events. PyTorch and GeNN own different CUDA
    # runtimes in this worker, so torch.cuda.synchronize() is not a valid
    # completion barrier for GeNN kernels.
    model.timing_enabled = True
    pop = model.add_neuron_population(
        "neurons",
        case.n_neuron,
        neuron_model,
        {
            "tau_mem": case.tau_mem,
            "c_m": case.c_m,
            "v_reset": case.v_reset,
            "v_threshold": case.v_threshold,
            "reset_delta": case.v_threshold - case.v_reset,
            "psc_decay": float(np.exp(-case.dt / case.tau_syn)),
        },
        {"V": 0.0, "PSC": 0.0},
    )
    pop.extra_global_params["input"].set_init_values(
        np.asarray(arrays["x"], dtype=np.float32).reshape(input_size)
    )
    synapses = model.add_synapse_population(
        "recurrent",
        "SPARSE",
        pop,
        pop,
        init_weight_update(
            "StaticPulse",
            {},
            {"g": np.asarray(arrays["values"], dtype=np.float32)},
        ),
        init_postsynaptic("DeltaCurr"),
    )
    synapses.set_sparse_connections(
        _source_indices(arrays["indptr"]).astype(np.uint32),
        np.asarray(arrays["indices"], dtype=np.uint32),
    )
    # FlyWire has a heavy-tailed out-degree distribution. GeNN's default
    # postsynaptic parallelism sizes every sparse row to the global maximum
    # out-degree, wasting almost all threads for this graph. Presynaptic
    # parallelism assigns one thread to each spiking source and walks only
    # that source's actual row.
    synapses.parallelism_hint = ParallelismHint.PRESYNAPTIC
    pop.spike_recording_enabled = record
    build_start = time.perf_counter()
    _record_phase(project.parent, f"{project.name}_code_generation_and_build")
    model.build(path_to_model=str(project))
    build_time = time.perf_counter() - build_start
    _record_phase(project.parent, f"{project.name}_load")
    if record:
        model.load(num_recording_timesteps=case.t_steps)
    else:
        model.load()
    _record_phase(project.parent, f"{project.name}_ready")
    return model, pop, build_time


def _genn_simulate(model, case: BenchCase) -> float:
    """Run one GeNN window and return native CUDA kernel time."""

    start = (
        model.neuron_update_time
        + model.presynaptic_update_time
        + model.postsynaptic_update_time
        + model.synapse_dynamics_time
    )
    for _ in range(case.t_steps):
        model.step_time()
    # Reading GeNN's counters synchronizes its own runtime. A
    # torch.cuda.synchronize() would only synchronize PyTorch's context.
    end = (
        model.neuron_update_time
        + model.presynaptic_update_time
        + model.postsynaptic_update_time
        + model.synapse_dynamics_time
    )
    return (end - start) * 1e3


def _run_genn(request: dict, arrays: dict[str, np.ndarray]) -> dict:
    """Execute the GeNN benchmark worker."""

    import pygenn

    case = BenchCase(**request["case"])
    work = Path(request["work"])
    options = request["provider_options"]
    _record_phase(work, "genn_performance_build")
    model, _, build_time = _create_genn_model(
        arrays,
        case,
        work / "genn_performance",
        record=False,
        options=options,
    )
    samples = []
    try:
        _record_phase(work, "genn_warmup_and_timing")
        for index in range(request["warmup"] + request["repeat"]):
            if index < request["warmup"]:
                phase = (
                    f"genn_warmup_{index + 1}_of_"
                    f"{request['warmup']}"
                )
            else:
                phase = (
                    f"genn_timing_{index - request['warmup'] + 1}_of_"
                    f"{request['repeat']}"
                )
            _record_phase(work, phase)
            if index:
                model.unload()
                model.load()
            sample = _genn_simulate(model, case)
            if index >= request["warmup"]:
                samples.append(sample)
    finally:
        model.unload()

    if request["check_correctness"]:
        _record_phase(work, "genn_correctness_build")
        recorded, pop, correctness_build = _create_genn_model(
            arrays,
            case,
            work / "genn_correctness",
            record=True,
            options=options,
        )
        build_time += correctness_build
        try:
            _genn_simulate(recorded, case)
            recorded.pull_recording_buffers_from_device()
            pop.vars["V"].pull_from_device()
            pop.vars["PSC"].pull_from_device()
            spikes = np.zeros(
                (case.t_steps, case.batch_size, case.n_neuron),
                dtype=np.float32,
            )
            for batch, (spike_times, spike_ids) in enumerate(
                pop.spike_recording_data
            ):
                timestep = np.rint(
                    np.asarray(spike_times) / case.dt
                ).astype(np.int64)
                ids = np.asarray(spike_ids, dtype=np.int64)
                valid = (timestep >= 0) & (timestep < case.t_steps)
                spikes[timestep[valid], batch, ids[valid]] = 1.0
            v = np.asarray(
                pop.vars["V"].current_values,
                dtype=np.float32,
            ).reshape(case.batch_size, case.n_neuron)
            psc = np.asarray(
                pop.vars["PSC"].current_values,
                dtype=np.float32,
            ).reshape(case.batch_size, case.n_neuron)
            # GeNN delivers spikes from the last simulated timestep into its
            # postsynaptic input buffer for consumption on the next neuron
            # update. Materialize that pending update so the reported final
            # PSC matches the benchmark's after-step state contract.
            source = _source_indices(arrays["indptr"])
            target = np.asarray(arrays["indices"], dtype=np.int64)
            value = np.asarray(arrays["values"], dtype=np.float32)
            recurrent = np.zeros_like(psc)
            for batch in range(case.batch_size):
                np.add.at(
                    recurrent[batch],
                    target,
                    spikes[-1, batch, source] * value,
                )
            psc = (
                np.float32(np.exp(-case.dt / case.tau_syn)) * psc
                + recurrent
            )
            np.savez(work / "correctness.npz", spikes=spikes, v=v, psc=psc)
        finally:
            recorded.unload()
    return {
        "latency_samples_ms": samples,
        "framework_version": f"PyGeNN {getattr(pygenn, '__version__', 'unknown')}",
        "build_time_s": build_time,
        "timing_method": (
            "genn_native_cuda_event_kernel_sum"
            ";sparse_parallelism=presynaptic"
            f";optimize_code={options['optimize_code']}"
        ),
    }


def _worker(request_path: Path) -> None:
    """Run an external framework worker and write machine-readable results."""

    request = json.loads(request_path.read_text())
    work = Path(request["work"])
    _record_phase(work, "worker_loading_case")
    loaded = np.load(work / "case.npz")
    arrays = {name: loaded[name] for name in loaded.files}
    _record_phase(work, "worker_dispatch")
    if request["provider"] == "brian2cuda":
        result = _run_brian2cuda(request, arrays)
    elif request["provider"] == "genn":
        result = _run_genn(request, arrays)
    else:
        raise ValueError(request["provider"])
    _record_phase(work, "worker_writing_result")
    (work / "result.json").write_text(json.dumps(result))
    _record_phase(work, "worker_complete")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: python -m benchmark.external_rsnn_simulators REQUEST")
    _worker(Path(sys.argv[1]))
