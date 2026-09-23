from __future__ import annotations

import argparse
import itertools
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np


@dataclass(frozen=True)
class RechunkResult:
    source_path: Path
    destination_path: Path
    array_path: str
    shape: tuple[int, ...]
    source_chunks: tuple[int, ...]
    destination_chunks: tuple[int, ...]
    chunks_processed: int
    chunks_written: int


def rechunk_ome_zarr_mask(
    source_path: str | Path,
    destination_path: str | Path,
    *,
    array_path: str = "s0",
    xy_chunk: int = 1024,
    progress: Callable[[int, int], None] | None = None,
) -> RechunkResult:
    """Copy a 2D/3D OME-Zarr mask into a new store with smaller XY chunks.

    The source store is opened read-only and is never modified. Existing
    destinations are rejected so an interrupted or mistaken conversion cannot
    overwrite the only copy of a mask.
    """

    try:
        import zarr
    except Exception as exc:
        raise RuntimeError("Mask rechunking requires the 'zarr' package.") from exc

    source_path = Path(source_path).expanduser()
    destination_path = Path(destination_path).expanduser()
    normalized_path = str(array_path).strip().strip("/")
    if not normalized_path:
        raise ValueError("array_path must identify an array inside the OME-Zarr store.")
    if int(xy_chunk) <= 0:
        raise ValueError("xy_chunk must be greater than zero.")
    if not source_path.exists():
        raise FileNotFoundError(source_path)
    if destination_path.exists():
        raise FileExistsError(
            f"Destination already exists: {destination_path}. "
            "Choose a new path; the rechunker never overwrites mask data."
        )

    source_root = zarr.open_group(str(source_path), mode="r")
    try:
        source = source_root[normalized_path]
    except KeyError as exc:
        raise ValueError(
            f"OME-Zarr mask array '{normalized_path}' was not found in {source_path}."
        ) from exc

    shape = tuple(int(value) for value in source.shape)
    if len(shape) not in (2, 3):
        raise ValueError(f"Mask rechunking supports 2D or 3D arrays; got shape {shape}.")
    source_chunks = _normalized_chunks(source, shape)
    destination_chunks = list(source_chunks)
    destination_chunks[-2] = min(shape[-2], int(xy_chunk))
    destination_chunks[-1] = min(shape[-1], int(xy_chunk))
    if len(shape) == 3:
        destination_chunks[0] = min(shape[0], max(1, source_chunks[0]))
    destination_chunks_tuple = tuple(destination_chunks)

    destination_root = zarr.open_group(str(destination_path), mode="w")
    _copy_attrs(source_root, destination_root)
    destination_parent = destination_root
    source_parent = source_root
    parts = normalized_path.split("/")
    for part in parts[:-1]:
        source_parent = source_parent[part]
        destination_parent = destination_parent.require_group(part)
        _copy_attrs(source_parent, destination_parent)

    destination = destination_parent.create_array(
        parts[-1],
        shape=shape,
        chunks=destination_chunks_tuple,
        dtype=np.dtype(source.dtype),
        fill_value=getattr(source, "fill_value", 0),
    )
    _copy_attrs(source, destination)
    destination_root.attrs["sam3_rechunk"] = {
        "complete": False,
        "source": str(source_path),
        "array_path": normalized_path,
        "source_chunks": list(source_chunks),
        "destination_chunks": list(destination_chunks_tuple),
    }

    grid_shape = tuple(
        (size + chunk - 1) // chunk
        for size, chunk in zip(shape, source_chunks, strict=True)
    )
    total = int(np.prod(grid_shape, dtype=np.int64))
    processed = 0
    written = 0
    fill_value = getattr(source, "fill_value", 0)

    for chunk_index in itertools.product(*(range(count) for count in grid_shape)):
        region = tuple(
            slice(index * chunk, min(size, (index + 1) * chunk))
            for index, chunk, size in zip(
                chunk_index, source_chunks, shape, strict=True
            )
        )
        block = np.asarray(source[region])
        processed += 1
        if block.size and not np.all(block == fill_value):
            destination[region] = block
            written += 1
        if progress is not None:
            progress(processed, total)

    destination_root.attrs["sam3_rechunk"] = {
        "complete": True,
        "source": str(source_path),
        "array_path": normalized_path,
        "source_chunks": list(source_chunks),
        "destination_chunks": list(destination_chunks_tuple),
    }
    return RechunkResult(
        source_path=source_path,
        destination_path=destination_path,
        array_path=normalized_path,
        shape=shape,
        source_chunks=source_chunks,
        destination_chunks=destination_chunks_tuple,
        chunks_processed=processed,
        chunks_written=written,
    )


def _normalized_chunks(array: Any, shape: tuple[int, ...]) -> tuple[int, ...]:
    chunks = getattr(array, "chunks", None)
    if chunks is None:
        return shape
    normalized = tuple(max(1, int(value)) for value in chunks)
    if len(normalized) != len(shape):
        raise ValueError(
            f"Source chunk rank {len(normalized)} does not match shape rank {len(shape)}."
        )
    return tuple(
        min(size, chunk)
        for size, chunk in zip(shape, normalized, strict=True)
    )


def _copy_attrs(source: Any, destination: Any) -> None:
    attrs = getattr(source, "attrs", {})
    if hasattr(attrs, "asdict"):
        payload = attrs.asdict()
    else:
        payload = dict(attrs)
    if payload:
        destination.attrs.update(payload)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Safely copy an OME-Zarr mask to a new store with smaller XY chunks."
    )
    parser.add_argument("source")
    parser.add_argument("destination")
    parser.add_argument("--array-path", default="s0")
    parser.add_argument("--xy-chunk", type=int, default=1024)
    args = parser.parse_args(argv)

    last_percent = -1

    def report(done: int, total: int) -> None:
        nonlocal last_percent
        percent = int(done * 100 / max(1, total))
        if percent >= last_percent + 5 or done == total:
            print(f"Rechunking: {done}/{total} source chunks ({percent}%)", flush=True)
            last_percent = percent

    result = rechunk_ome_zarr_mask(
        args.source,
        args.destination,
        array_path=args.array_path,
        xy_chunk=args.xy_chunk,
        progress=report,
    )
    print(
        f"Created {result.destination_path} /{result.array_path}: "
        f"shape={result.shape}, chunks={result.destination_chunks}, "
        f"nonempty source chunks={result.chunks_written}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
