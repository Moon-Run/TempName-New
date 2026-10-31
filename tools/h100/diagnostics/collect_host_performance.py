#!/usr/bin/env python3
"""Read-only host/GPU evidence collection; no Torch import or benchmark load."""

import argparse
import datetime
import json
import math
import os
from pathlib import Path, PurePosixPath
import platform
import re
import socket
import subprocess


CGROUP_FILES = (
    "cpu.max", "cpu.max.burst", "cpu.weight", "cpu.stat", "cpu.pressure",
    "cpuset.cpus", "cpuset.cpus.effective", "cpuset.mems",
    "cpuset.mems.effective", "cpu.cfs_quota_us", "cpu.cfs_period_us",
    "cpu.shares", "cpuacct.usage", "cpuset.effective_cpus",
    "cpuset.effective_mems",
)


def read_text(path):
    try:
        return {"text": Path(path).read_text().strip()}
    except (OSError, UnicodeError) as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}


def run_command(argv, timeout):
    try:
        result = subprocess.run(argv, capture_output=True, text=True,
                                errors="replace", timeout=timeout, check=False)
        return {"argv": argv, "returncode": result.returncode,
                "stdout": result.stdout, "stderr": result.stderr}
    except subprocess.TimeoutExpired as exc:
        def as_text(value):
            return value.decode(errors="replace") if isinstance(value, bytes) else value
        return {"argv": argv, "error": f"timeout after {timeout}s",
                "stdout": as_text(exc.stdout), "stderr": as_text(exc.stderr)}
    except OSError as exc:
        return {"argv": argv, "error": f"{type(exc).__name__}: {exc}"}


def unescape_mount(value):
    return re.sub(r"\\([0-7]{3})", lambda m: chr(int(m[1], 8)), value)


def cgroup_evidence(pid):
    """Inspect each accessible ancestor: a parent's quota may constrain a child."""
    membership = read_text(f"/proc/{pid}/cgroup")
    mountinfo = read_text(f"/proc/{pid}/mountinfo")
    result = {"membership": membership, "controllers": [],
              "visibility": "Only ancestors visible in this PID's mount namespace are accessible."}
    if "error" in mountinfo:
        result["mountinfo_error"] = mountinfo["error"]
    if "text" not in membership or "text" not in mountinfo:
        return result
    mounts = []
    for line in mountinfo["text"].splitlines():
        left, sep, right = line.partition(" - ")
        if not sep:
            continue
        fields, tail = left.split(), right.split()
        if len(fields) >= 5 and len(tail) >= 3 and tail[0] in ("cgroup", "cgroup2"):
            mounts.append((tail[0], unescape_mount(fields[3]),
                           unescape_mount(fields[4]), set(tail[2].split(","))))
    for line in membership["text"].splitlines():
        fields = line.split(":", 2)
        if len(fields) != 3:
            continue
        hierarchy, names, member_path = fields
        controllers = set(names.split(","))
        is_v2 = hierarchy == "0" and not names
        if not is_v2 and not controllers.intersection({"cpu", "cpuacct", "cpuset"}):
            continue
        matched = False
        for kind, root, mountpoint, options in mounts:
            if kind != ("cgroup2" if is_v2 else "cgroup"):
                continue
            if not is_v2 and not controllers.intersection(options):
                continue
            matched = True
            entry = {"version": 2 if is_v2 else 1, "controllers": names,
                     "membership_path": member_path, "mount_root": root,
                     "mountpoint": mountpoint, "ancestors": []}
            result["controllers"].append(entry)
            try:
                relative = PurePosixPath(member_path).relative_to(PurePosixPath(root))
            except ValueError:
                entry["error"] = "Membership is outside the visible mount root; cannot resolve safely."
                continue
            if ".." in relative.parts:
                entry["error"] = "Membership contains parent traversal; not followed."
                continue
            base = Path(f"/proc/{pid}/root") / mountpoint.lstrip("/")
            current = base.joinpath(*relative.parts)
            for _ in range(64):
                values = {}
                for name in CGROUP_FILES:
                    path = current / name
                    # Keep permission errors, but absent v1/v2-only fields add no evidence.
                    value = read_text(path)
                    if not value.get("error", "").startswith("FileNotFoundError:"):
                        values[name] = value
                entry["ancestors"].append({"path": str(current), "files": values})
                if current == base:
                    break
                current = current.parent
            else:
                entry["truncated"] = "More than 64 ancestors"
        if not matched:
            result["controllers"].append({"controllers": names,
                                          "error": "No matching visible cgroup mount"})
    return result


def process_evidence(pid):
    status = read_text(f"/proc/{pid}/status")
    result = {"pid": pid, "status": status}
    if "text" in status:
        keep = {"Name", "State", "Threads", "Cpus_allowed_list", "Mems_allowed_list"}
        result["status"] = {key: value.strip() for line in status["text"].splitlines()
                            for key, _, value in [line.partition(":")] if key in keep}
    try:
        result["affinity_cpus"] = sorted(os.sched_getaffinity(pid))
    except (OSError, AttributeError) as exc:
        result["affinity_error"] = f"{type(exc).__name__}: {exc}"
    result["cgroups"] = cgroup_evidence(pid)
    result["thread_cpu_sets"] = {}
    try:
        for task in sorted(Path(f"/proc/{pid}/task").iterdir()):
            status = read_text(task / "status")
            if "text" in status:
                status = {key: value.strip() for line in status["text"].splitlines()
                          for key, _, value in [line.partition(":")]
                          if key in {"Cpus_allowed_list", "Mems_allowed_list"}}
            result["thread_cpu_sets"][task.name] = status
    except OSError as exc:
        result["thread_cpu_sets_error"] = str(exc)
    return result


def legacy_rank_binding():
    result = {"world_size": 8,
              "formula": "allowed_cpus[local_rank * len(allowed_cpus) // 8]",
              "note": "Historical singleton policy only; current workers inherit the launcher CPU set. "
                      "This comparison uses collector affinity, which may differ from the launcher.",
              "ranks": []}
    try:
        cpus = sorted(os.sched_getaffinity(0))
        result["launcher_allowed_cpus"] = cpus
        for rank in range(8):
            cpu = cpus[rank * len(cpus) // 8]
            base = Path(f"/sys/devices/system/cpu/cpu{cpu}")
            result["ranks"].append({"rank": rank, "cpu": cpu,
                "topology": {name: read_text(base / "topology" / name) for name in
                             ("core_id", "physical_package_id", "thread_siblings_list")},
                "numa_nodes": [path.name for path in sorted(base.glob("node[0-9]*"))]})
    except (OSError, AttributeError, IndexError) as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    return result


def cpu_evidence():
    result = {"online": read_text("/sys/devices/system/cpu/online"),
              "loadavg": read_text("/proc/loadavg"),
              "pressure": read_text("/proc/pressure/cpu"), "frequency_policies": {}}
    info = read_text("/proc/cpuinfo")
    if "text" in info:
        wanted = {"processor", "model name", "cpu MHz", "physical id", "core id"}
        result["cpuinfo"] = [{key.strip(): value.strip()
                              for line in section.splitlines() if ":" in line
                              for key, value in [line.split(":", 1)] if key.strip() in wanted}
                             for section in info["text"].split("\n\n")]
    else:
        result["cpuinfo"] = info
    try:
        policies = sorted(Path("/sys/devices/system/cpu/cpufreq").glob("policy*"))
        for policy in policies:
            result["frequency_policies"][policy.name] = {
                name: read_text(policy / name) for name in
                ("related_cpus", "scaling_driver", "scaling_governor", "scaling_cur_freq",
                 "scaling_min_freq", "scaling_max_freq", "cpuinfo_min_freq", "cpuinfo_max_freq")}
        if not policies:
            result["frequency_note"] = "No cpufreq policy directories visible."
    except OSError as exc:
        result["frequency_error"] = str(exc)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path,
                        help="New output directory (must not already exist)")
    parser.add_argument("--pid", action="append", type=int, default=[],
                        help="Inspect a worker PID's affinity and cgroups; repeat for each rank")
    parser.add_argument("--timeout", type=float, default=15.0,
                        help="Timeout in seconds for each read-only command (default: 15)")
    args = parser.parse_args()
    if not math.isfinite(args.timeout) or args.timeout <= 0 or any(pid <= 0 for pid in args.pid):
        parser.error("--timeout and each --pid must be positive")
    try:
        args.output.mkdir(parents=True, exist_ok=False)
    except OSError as exc:
        parser.error(f"Cannot create new output directory: {exc}")
    report = {"schema_version": 1,
              "collected_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
              "hostname": socket.gethostname(), "platform": platform.platform(),
              "python_version": platform.python_version(), "collector_pid": os.getpid(),
              "cpu": cpu_evidence(),
              "worker_cpu_affinity_policy": "inherit",
              "legacy_single_cpu_binding": legacy_rank_binding(),
              "processes": [process_evidence(pid) for pid in dict.fromkeys([os.getpid(), *args.pid])],
              "commands": {},
              "limitations": ["One snapshot cannot establish full-run exclusivity or clock stability.",
                              "No environment dump, process command lines, settings changes or GPU load.",
                              "Inaccessible host-level cgroup ancestors may impose additional limits."]}
    commands = {
        "lscpu": ["lscpu", "--json"], "numa": ["numactl", "--hardware"],
        "gpu_details": ["nvidia-smi", "-q"],
        "gpu_topology": ["nvidia-smi", "topo", "-m"],
        "nvlink_status": ["nvidia-smi", "nvlink", "-s"],
        "gpu_processes": ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,used_gpu_memory",
                          "--format=csv,noheader"],
    }
    for name, command in commands.items():
        report["commands"][name] = run_command(command, args.timeout)
    path = args.output / "host-performance.json"
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(path)


if __name__ == "__main__":
    main()
