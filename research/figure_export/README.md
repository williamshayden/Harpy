# Export retained experiment data for publication

`export_figures.py` derives compact plotting and replay data from the frozen
spectrum-preservation study and a separately labeled qualification diagnostic.
It does not train models or evaluate new episodes. This research utility is not
part of the installed package's public API.

From the revised Harpy checkout with Python 3.12 and the base dependencies:

```bash
python research/figure_export/export_figures.py --output outputs/figure-data
```

The command requires these retained local archives, whose hashes are pinned in
the script:

- `outputs/spectrum-preservation-study/experiment.json`
- `outputs/v1-respec-qualification/spectrum-failure-case.json`

They have not been published, so a fresh clone cannot reproduce this export on
its own. The script also checks the frozen study's research source hashes,
recomputes all metrics through Harpy's strict result reader, and verifies matched
initial observations. Reconstructed captures must match saved waveform, spectrum,
and realized-SNR evidence. Replay states are derived from saved actions; they are
not captured per-step observations or new inferences.

The destination must be new. `data-provenance.json` is written last to mark a
complete export. Preserve interrupted output and retry with a new destination.

Blog copy, site layout, the Matplotlib renderer, and rendered assets are maintained
outside the Harpy repository. The exporter remains here because reproducible data
derivation belongs beside the experiment implementation and evidence. User-facing
instructions live in the [canonical guide](../../docs/v1-getting-started.md).
