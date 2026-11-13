"""Disk-backed, sequential ROI experiments. No Qt or model dependencies."""
from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from threading import Event
import time
from typing import Callable

import numpy as np
import zarr


class ExperimentCancelled(Exception):
    pass


@dataclass(frozen=True)
class ExperimentPlan:
    sizes: tuple[int, ...]
    modes: tuple[str, ...]
    overlap: float
    merge_seams: bool
    save_tiles: bool = False

    def __post_init__(self):
        if not self.sizes or len(set(self.sizes)) != len(self.sizes) or any(s <= 0 for s in self.sizes):
            raise ValueError("Choose unique positive ROI sizes.")
        if not self.modes or len(set(self.modes)) != len(self.modes) or any(m not in ("encoded", "current", "reference") for m in self.modes):
            raise ValueError("Choose encoded reference or a legacy comparison series.")
        if not 0 <= self.overlap <= .5:
            raise ValueError("Overlap must be between 0 and 50%.")


def write_json(path: Path, payload):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(payload, indent=2, default=str), encoding='utf-8')
    temporary.replace(path)


def check_cancel(cancel: Event):
    if cancel.is_set():
        raise ExperimentCancelled()


def create_mask(path: Path, shape):
    group = zarr.open_group(str(path), mode='w')
    array = group.create_array('s0', shape=tuple(shape), chunks=(1024, 1024), dtype='uint32', fill_value=0)
    return group, array


def blocks(shape, side=1024):
    for y in range(0, shape[0], side):
        for x in range(0, shape[1], side):
            yield (slice(y, min(y+side, shape[0])), slice(x, min(x+side, shape[1])))


def finish_mask(group, cancel: Event):
    """Build first-nonzero label levels and exact instance statistics in chunks."""
    array = group['s0']
    counts: dict[int, int] = {}
    for region in blocks(array.shape):
        check_cancel(cancel)
        ids, numbers = np.unique(np.asarray(array[region]), return_counts=True)
        for label, number in zip(ids, numbers):
            if label:
                key = int(label)
                counts[key] = counts.get(key, 0) + int(number)
    datasets = [{'path': 's0', 'coordinateTransformations': [{'type': 'scale', 'scale': [1., 1.]}]}]
    parent = array
    level = 0
    while max(parent.shape) > 1024:
        level += 1
        shape = tuple((s+1)//2 for s in parent.shape)
        child = group.create_array(f's{level}', shape=shape, chunks=(1024, 1024), dtype='uint32', fill_value=0)
        for ys, xs in blocks(shape):
            check_cancel(cancel)
            source = np.asarray(parent[ys.start*2:min(ys.stop*2,parent.shape[0]), xs.start*2:min(xs.stop*2,parent.shape[1])])
            reduced = np.zeros((ys.stop-ys.start, xs.stop-xs.start), dtype=np.uint32)
            for dy, dx in ((0,0), (0,1), (1,0), (1,1)):
                part = source[dy::2, dx::2]
                view = reduced[:part.shape[0], :part.shape[1]]
                np.copyto(view, part, where=view == 0)
            child[ys,xs] = reduced
        datasets.append({'path': f's{level}', 'coordinateTransformations': [{'type': 'scale', 'scale': [float(2**level)]*2}]})
        parent = child
    group.attrs['multiscales'] = [{'version': '0.4', 'axes': [{'name':'y','type':'space'}, {'name':'x','type':'space'}], 'datasets': datasets}]
    total = int(np.prod(array.shape))
    labeled = sum(counts.values())
    return {'instances': len(counts), 'labeled_pixels': labeled,
            'coverage': labeled/total, 'largest_instance_pixels': max(counts.values(), default=0)}


def merge_mask(source, destination, tiles, vertical_pairs, horizontal_pairs, cancel: Event):
    """Apply the existing contact-based seam policy without a full RAM array."""
    parent = {}
    def find(value):
        parent.setdefault(value, value)
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value
    def collect(pairs):
        for (a,b), count in pairs.items():
            if count >= 8:
                ra, rb = find(a), find(b)
                if ra != rb:
                    parent[max(ra,rb)] = min(ra,rb)
    xs = sorted({v for b in tiles for v in (b.x0,b.x1) if 0 < v < source.shape[1]})
    ys = sorted({v for b in tiles for v in (b.y0,b.y1) if 0 < v < source.shape[0]})
    for x in xs:
        check_cancel(cancel)
        collect(vertical_pairs(source,x,2))
    for y in ys:
        check_cancel(cancel)
        collect(horizontal_pairs(source,y,2))
    # Match normal scanning's numeric ID ordering, including untouched labels.
    present = set()
    for region in blocks(source.shape):
        check_cancel(cancel)
        present.update(int(v) for v in np.unique(source[region]) if v)
    root_counts = {}
    for label in present:
        root = find(label)
        root_counts[root] = root_counts.get(root,0)+1
    groups = sum(n > 1 for n in root_counts.values())
    if not groups:
        for region in blocks(source.shape):
            check_cancel(cancel)
            destination[region] = source[region]
        return 0
    root_to_output = {}
    mapping = np.zeros(max(present, default=0)+1, dtype=np.uint32)
    for label in sorted(present):
        root = find(label)
        if root not in root_to_output:
            root_to_output[root] = len(root_to_output)+1
        mapping[label] = root_to_output[root]
    for region in blocks(source.shape):
        check_cancel(cancel)
        destination[region] = mapping[np.asarray(source[region])]
    return groups


def run_experiment(*, folder: Path, plan: ExperimentPlan, shape, constants: dict,
                   tile_plan: Callable, infer: Callable, compose: Callable,
                   vertical_pairs: Callable, horizontal_pairs: Callable,
                   cancel: Event, progress: Callable = lambda message: None):
    """Infer returns (labels, boxes, scores, path-name, extra-metadata)."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    manifest = {'format': 'sam3-batch-experiment-v1', 'status': 'running',
                'variable': 'ROI size', 'plan': asdict(plan), 'constants': constants, 'runs': []}
    write_json(folder/'experiment.json', manifest)
    try:
        for mode in plan.modes:
            for size in plan.sizes:
                check_cancel(cancel)
                name = f'{mode}_roi_{size}'
                directory = folder/name
                directory.mkdir(exist_ok=False)
                record = {'name':name, 'mode':mode, 'roi_size':size, 'status':'running',
                          'tiles_completed':0, 'empty_tiles':0, 'natural_tiles':0, 'reference_tiles':0,
                          'encoded_tiles':0,'skipped_uniform_tiles':0}
                manifest['runs'].append(record)
                write_json(folder/'experiment.json', manifest)
                started = time.monotonic()
                tiles = tile_plan(shape, (size,size), plan.overlap)
                record['tile_count'] = len(tiles)
                group, array = create_mask(directory/'composed.ome.zarr',shape)
                next_id = 1
                if plan.save_tiles:
                    (directory/'tiles').mkdir()
                with (directory/'tiles.jsonl').open('w',encoding='utf-8') as log:
                    for index,bounds in enumerate(tiles):
                        check_cancel(cancel)
                        progress(f'{name}: tile {index+1}/{len(tiles)}')
                        tick = time.monotonic()
                        labels,boxes,scores,path,extra = infer(bounds,mode)
                        check_cancel(cancel)
                        if labels.shape != (bounds.height,bounds.width):
                            raise ValueError('Inference returned the wrong target tile shape.')
                        region = (slice(bounds.y0,bounds.y1),slice(bounds.x0,bounds.x1))
                        patch = np.asarray(array[region])
                        # Preserve normal scanning's first-nonzero composition.
                        next_id = compose(patch,labels,next_id)
                        array[region] = patch
                        empty = not np.any(labels)
                        record['tiles_completed'] += 1
                        record['empty_tiles'] += int(empty)
                        record[f'{path}_tiles'] += 1
                        uniform=bool(extra.get('skipped_uniform_tile'))
                        record['skipped_uniform_tiles'] += int(uniform)
                        entry = {'tile':index+1,'bounds_yxyx':[bounds.y0,bounds.x0,bounds.y1,bounds.x1],
                                 'prompt_path':path,'state':'skipped-uniform' if uniform else ('processed-empty' if empty else 'processed-with-candidates'),
                                 'seconds':time.monotonic()-tick, 'target_boxes':np.asarray(boxes).tolist(),
                                 'scores':[] if scores is None else np.asarray(scores).tolist(), **extra}
                        log.write(json.dumps(entry)+'\n');log.flush()
                        if plan.save_tiles:
                            np.savez_compressed(directory/'tiles'/f'{index+1:06d}.npz',labels=labels,boxes=boxes,
                                                scores=np.asarray([]) if scores is None else scores)
                progress(f'{name}: summarizing composed result')
                record['composed_stats'] = finish_mask(group,cancel)
                record['composed_path'] = str((directory/'composed.ome.zarr').relative_to(folder))
                record['final_path'] = record['composed_path']
                record['seam_merge_groups'] = 0
                if plan.merge_seams:
                    progress(f'{name}: merging seams')
                    final_group, final_array = create_mask(directory/'final.ome.zarr',shape)
                    record['seam_merge_groups'] = merge_mask(array,final_array,tiles,vertical_pairs,horizontal_pairs,cancel)
                    record['final_stats'] = finish_mask(final_group,cancel)
                    record['final_path'] = str((directory/'final.ome.zarr').relative_to(folder))
                else:
                    record['final_stats'] = record['composed_stats']
                record.update(status='complete', seconds=time.monotonic()-started)
                write_json(folder/'experiment.json',manifest)
                write_comparison(folder,manifest)
                progress(f'Completed {name}')
        manifest['status'] = 'complete'
    except ExperimentCancelled:
        manifest['status'] = 'cancelled'
        if manifest['runs'] and manifest['runs'][-1]['status']=='running':
            manifest['runs'][-1]['status'] = 'cancelled'
    except Exception as exc:
        manifest['status'] = 'failed'
        manifest['error'] = str(exc)
        if manifest['runs'] and manifest['runs'][-1]['status']=='running':
            manifest['runs'][-1].update(status='failed',error=str(exc))
    finally:
        write_json(folder/'experiment.json',manifest)
        write_comparison(folder,manifest)
    return manifest


def write_comparison(folder,manifest):
    fields = ['name','mode','roi_size','status','seconds','tile_count','tiles_completed','empty_tiles',
              'natural_tiles','reference_tiles','encoded_tiles','skipped_uniform_tiles','instances','coverage','largest_instance_pixels','seam_merge_groups']
    with (folder/'comparison.csv').open('w',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields,extrasaction='ignore');writer.writeheader()
        for run in manifest['runs']:
            writer.writerow({**run,**run.get('final_stats',{})})
