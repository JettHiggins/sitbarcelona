# Submission candidate

`sitbarcelona.csv` is a local copy of the recommended Mall trajectory-model
snapshot. Rename it and `technical-document.md` to the team name used in the
registration form before upload. The CSV is Git-ignored; its identical source
is tracked at `possession-viewer/data/mall-model.csv`.

Recreate the candidate from the repository root without training:

```sh
python -c "from shutil import copyfile; copyfile('possession-viewer/data/mall-model.csv', 'submission/sitbarcelona.csv')"
pipeline/.venv/bin/python -m pipeline.compare submission/sitbarcelona.csv
```

Use the official challenge checkout's `evaluation/validate_submission.py` on
the exact upload file too. This candidate passed that validator: 199,319 ordered
rows, 40,462 X frames outside the halves and no abstentions inside them.
`manifest.json` records its source and checksum.

The draft technical document explains the algorithm and its limitations.
The author's production checkpoints, input/model/output hashes and pinned
production requirements are now bundled. Follow
`pipeline/mall/production/README.md` for inference from raw inputs without
benchmark downloads or training. The author reports byte-identical reproduction
from a clean worktree in the recorded Windows/CUDA environment.
An independent replay of the merged code on Linux ARM64 CPU also regenerated
the candidate byte for byte from raw inputs with an initially empty cache.
See `reproduction-verification.json` and `cpu-verification-environment.txt`.

The submission form also needs the public repository URL. Local integration
commits must be pushed before that URL contains this work. No form submission
has been made by this workflow.
