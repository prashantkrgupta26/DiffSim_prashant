# scripts/remote — the remote-deploy toolkit

The **canonical policy and workflow** live in **`docs/dev/remote-workflow.md`**
(spec: `docs/dev/specs/2026-07-17-remote-deploy-workflow-design.md`). Read that first.

TL;DR: the **office Mac is the brain and its DiffSim repo is master (source of
truth)**; **gpubox** and **nova** are compute workers. Code flows Mac→worker;
results+commits flow worker→Mac and are reconciled into master. The Mac is the
sole GitHub gatekeeper and never runs the heavy GPU solves.

Scripts here: `config.sh` (shared config), `gpubox-{sync,run,dispatch,poll,fetch,test}.sh`,
`nova-{sync-submit,poll}.sh`, `remote-doctor.sh`, `lib.sh`.
