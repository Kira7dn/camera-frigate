"""Shared Frigate recording selection and clip materialization."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

from pathvalidate import sanitize_filename

from frigate.infrastructure.config import FrigateConfig
from frigate.const import CACHE_DIR
from frigate.models import Recordings


def prepare_recording_clip(
    config: FrigateConfig,
    camera: str,
    start_ts: float,
    end_ts: float,
    *,
    output: str = "pipe:",
) -> tuple[Path, list[str]] | None:
    recordings = (
        Recordings.select(
            Recordings.path,
            Recordings.start_time,
            Recordings.end_time,
        )
        .where(
            (Recordings.start_time.between(start_ts, end_ts))
            | (Recordings.end_time.between(start_ts, end_ts))
            | ((start_ts > Recordings.start_time) & (end_ts < Recordings.end_time))
        )
        .where(Recordings.camera == camera)
        .order_by(Recordings.start_time.asc())
    )
    if recordings.count() == 0:
        return None
    file_name = sanitize_filename(f"playlist_{camera}_{start_ts}-{end_ts}.txt")
    if len(file_name) > 1000:
        raise ValueError("recording clip filename exceeded 1000 characters")
    playlist = Path(CACHE_DIR) / file_name
    playlist.parent.mkdir(parents=True, exist_ok=True)
    with playlist.open("w", encoding="utf-8") as handle:
        for clip in recordings:
            handle.write(f"file '{clip.path}'\n")
            if clip.start_time < start_ts:
                handle.write(f"inpoint {int(start_ts - clip.start_time)}\n")
            if clip.end_time > end_ts:
                handle.write(f"outpoint {int(end_ts - clip.start_time)}\n")
    command = [
        config.ffmpeg.ffmpeg_path,
        "-hide_banner",
        "-y",
        "-protocol_whitelist",
        "pipe,file",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        os.fspath(playlist),
        "-c",
        "copy",
        "-movflags",
        "frag_keyframe+empty_moov",
        "-f",
        "mp4",
        output,
    ]
    return playlist, command


def materialize_recording_clip(
    config: FrigateConfig,
    camera: str,
    start_ts: float,
    end_ts: float,
    output: str | Path,
) -> bool:
    destination = Path(output)
    prepared = prepare_recording_clip(
        config, camera, start_ts, end_ts, output=os.fspath(destination)
    )
    if prepared is None:
        return False
    playlist, command = prepared
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    command[-1] = os.fspath(temporary)
    try:
        completed = subprocess.run(
            command,
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        if completed.returncode != 0:
            return False
        os.replace(temporary, destination)
        return True
    finally:
        playlist.unlink(missing_ok=True)
        temporary.unlink(missing_ok=True)
