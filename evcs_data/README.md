# EVCS Data Directory

This directory stores EVCS-related datasets used by `real_data_loader.py`.

## Primary dataset

ACN-Data (Caltech/JPL charging sessions):
- Official page: https://ev.caltech.edu/dataset
- Static mirror repo (format used by this project):
  - https://github.com/tongxin-li/ACN-Data-Static

Expected local structure (used by loader):

```text
evcs_data/
  ACN-Data-Static-main/
    time series data/
      ...
```

## Notes

- Keep large raw files out of Git history when possible.
- If data is unavailable, some parts of the pipeline can fall back to synthetic data.
