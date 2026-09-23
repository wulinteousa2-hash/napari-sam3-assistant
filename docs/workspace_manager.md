# SAM3 Workspace Manager

Open the independent dock widget from:

```text
Plugins > SAM3 Assistant: Workspace Manager
```

The manager keeps session metadata separate from large writable data:

- the `.sam3.json` manifest stores layer references, display state, and viewer state;
- writable Labels data stays in OME-Zarr;
- normal Save and Save As never snapshot an existing Zarr mask;
- Portable Snapshot is the explicit operation that copies referenced data.

## Commands

- **New** clears the viewer after confirmation and starts an unsaved workspace.
- **Open** loads a SAM3 manifest. Referenced Labels arrays are opened lazily with
  `r+` so edits write directly to their Zarr stores.
- **Save** atomically updates the current manifest. A new in-memory Labels layer
  is persisted once to a sibling `*_sam3_data` directory using
  `1024 x 1024` XY chunks and is rebound to that writable array.
- **Save As** creates another manifest that references the same durable data.
  It does not duplicate masks.
- **Open Recent** loads the selected path from SAM3 Assistant's own recent list.
- **Portable Snapshot** copies the manifest and every referenced local source
  into a selected empty folder. This may copy very large TIFF and Zarr data.

## Legacy Workspace Compatibility

Open also accepts the shared/Myelin version-2 `workspace.json` format. SAM3 maps
its `source.path` image references, `workspace_assets` OME-Zarr labels,
`asset_dataset` paths, Shapes, and Points into the SAM3 model without copying
mask pixels. Imported label arrays are opened in writable `r+` mode.

Opening a legacy manifest does not alter it. Use Save As with a
`*.sam3.json` name if the original legacy manifest must remain unchanged.

## Safety Rules

Workspace loading and saving never call `np.asarray()` on an entire Zarr mask.
Existing durable stores are never overwritten during ordinary Save. If a new
in-memory layer would collide with an existing store name, a new unique store is
created.

A workspace manifest is not a backup of externally referenced data. Use
Portable Snapshot when a self-contained archival copy is required.
