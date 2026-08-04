#!/usr/bin/env bash
# Submit one GH200 capacity-probe size. Args: NX NY NZ [JOBNAME] [EXTRA...]
# EXTRA passes through to gh200_probe.sh (e.g. "--managed" arg-style not
# supported here; use submit_probe_managed.sh for oversubscription).
NX=$1; NY=$2; NZ=$3; NAME=${4:-gh2p-${NX}}
sbatch -A mech-ai --partition=nova-arm --qos=normal --gres=gpu:gh200:1 \
  --cpus-per-task=8 --time=01:30:00 --mem=400G \
  --job-name="$NAME" \
  --output=cluster/results/gh200-probe2/${NAME}-%j.out \
  --wrap="bash cluster/gh200_probe.sh $NX $NY $NZ 4"
