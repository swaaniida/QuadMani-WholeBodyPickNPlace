# Existing AS2 training interruption investigation

All times below are Asia/Seoul (UTC+09:00), 2026-10-29.
Investigated read-only; no existing training code, checkpoint, process, permissions,
or server configuration was changed.

## Finding

The strongest supported explanation is VS Code terminal lifecycle failure:
client disconnect, terminal output backpressure blocking training, followed by
server shutdown and terminal cleanup. There is no recorded exit status or signal
for training PID 220646, so its exact final signal and death time are unconfirmed.
The extension-host exit code 0 belongs to PID 343182, not the training process.

## Evidence timeline

- 01:55:10.779: remoteagent.log line 39 records client disconnect and a 3-hour
  reconnection grace period. User supplied this otherwise unreadable vbc log.
- 01:58:27.884: last training console output and history step 7103, out of 45000.
  The local W&B binary contains 7104 history records and no exit/final/footer.
- Between disconnect and the last output: 105087 characters across 60 W&B
  output_raw records. Installed VS Code ptyHostMain.js pauses PTY reads when
  unacknowledged output exceeds 100000 characters. This closely matches the
  observed stall, but no thread stack or trace-level flow-control event survives.
- 04:57:38.412: W&B still records training-process RSS 8200.40625 MiB and
  19 threads; training was stalled before it disappeared, rather than dying at
  01:58. Last system/GPU stats timestamp is 04:57:53.409.
- 04:55:16.525: remoteagent.log line 41 records 3-hour grace expiry.
- 05:00:16.534: remoteagent.log line 42 records
  `ServerLifetime: all consumers inactive, shutting down`.
- Installed server-main.js calls terminal shutdownAll when disposing the PTY
  host. Installed ptyHostMain.js shutdown calls node-pty kill. This supports
  server cleanup as the explanation for loss of the terminal-attached process.

## Checks and limits

- Container PID 1 started October 26 and the Codex daemon survived: no container
  restart at the incident time.
- Cgroup memory.events and memory.events.local report oom=0 and oom_kill=0.
  Host kernel logs are inaccessible; these counters do not rule out every
  possible host-level failure.
- NVIDIA ECC counters are zero. This is not a complete historical GPU fault log.
- W&B output ends without a traceback or normal training completion message.
- vbc VS Code logs are inaccessible to suhwan; relevant excerpts were supplied
  by the user, with no permission changes.

## Recovery artifact

Last saved checkpoint: `low-level-as2piper/logs/as2piper-low/plate_only_C_original_20261028/model_7000.pt`.
Earlier checkpoints remain. No resume was started during this investigation.
For future long runs, detach process lifetime and stdout/stderr from the VS Code
terminal, e.g. an available tmux session or nohup with explicit file redirection.
Merely retaining a PID after disconnect is insufficient evidence of progress;
monitor iteration and checkpoint timestamps as well.
