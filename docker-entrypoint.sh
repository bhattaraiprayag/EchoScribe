#!/bin/sh
set -eu

cache_root="${MODELS_CACHE_DIR:-/app/models_cache}"

mkdir -p "$cache_root/whisper_models" "$cache_root/silero_vad"
chown -R appuser:appuser "$cache_root"

exec gosu appuser "$@"
