The central repository for our project for MoreThanAHack2026

To use the possession viewer
```
cd possession-viewer
python3 server.py
```

Open http://127.0.0.1:8081/ (the viewer's dedicated local address).

The independent [possession pipelines](pipeline/README.md) generate CSVs for the viewer.

Lenny, Mall, and the November 11 Mall experiments are combined on `main`.
Select `mall-referee-supervised.csv` in the viewer to inspect the new supplied
predictions. See the [integration notes](mall-improvement/INTEGRATION.md) for
verified scores, source history, and missing reproduction dependencies.
