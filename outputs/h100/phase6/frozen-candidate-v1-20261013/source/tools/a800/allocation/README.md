# Four-A800 holding allocation

`hold_tp4.sbatch` requests one `gpu_a800` node, four GPUs, eight CPUs and a
two-day wall-time limit. It follows `arrival8/allocation.sbatch`'s outer-loop
protocol with an independent control directory. Submit a frozen copy in a fresh
directory; retain the submission receipt and script hash there.

The batch process probes GPU resources and keeps a 30-second heartbeat under
`allocation-JOB_ID/heartbeat.txt`. A failed probe is recorded and the allocation
continues holding. Internal tests can run using `srun --jobid=JOB_ID`; completing
or failing a test does not terminate the outer batch process. No GPU work runs
unless an internal step is explicitly launched or `campaign-ready.sh` is staged.
The optional staged campaign runs once, and its exit status is preserved.

Creating `RELEASE` in the control directory ends the outer loop; do this only
when the user explicitly requests release. Slurm still enforces the two-day
limit. Do not cancel an allocation or terminate its outer process as part of
test cleanup.

User-authorized allocation on 2026-10-04: job **182708**, control directory
`logs/a800/allocations/a800-tp4-hold-20261004/`, running on `d1n41a15g01` from
2026-10-04 00:00:42 until 2026-10-06 00:00:42 Asia/Shanghai. Resource validation
confirmed four A800-SXM4-80GB GPUs with NV8 connectivity. Check Slurm for current
state; this entry is a submission record, not a promise of future availability.
