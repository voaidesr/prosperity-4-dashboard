# Prosperity Log Dashboard Deployment

This is the deployable upload-first Streamlit dashboard.

Entrypoint:

```text
dashboard/app.py
```

The app accepts `.log` and `.txt` Prosperity backtest logs through the browser,
parses them in memory, and does not write uploaded files to disk.

## Local Run

From the repository root:

```bash
pip install -r dashboard/requirements.txt
streamlit run dashboard/app.py
```

Optional password gate:

```bash
export DASHBOARD_PASSWORD='long-random-password'
streamlit run dashboard/app.py
```

## Docker Run

Build from the repository root:

```bash
docker build -f dashboard/Dockerfile -t prosperity-dashboard .
```

Run locally:

```bash
docker run --rm -p 8501:8501 \
  -e DASHBOARD_PASSWORD='long-random-password' \
  -e MAX_UPLOAD_MB=25 \
  prosperity-dashboard
```

Or use Compose:

```bash
cd dashboard
DASHBOARD_PASSWORD='long-random-password' docker compose up --build
```

## Streamlit Community Cloud

Recommended cheapest/easiest deployment:

1. Push this repo to GitHub.
2. Go to `share.streamlit.io` and create an app.
3. Select this repository and set the app path to `dashboard/app.py`.
4. Use Python 3.12.
5. Add this secret in Advanced settings:

```toml
dashboard_password = "long-random-password"
```

6. Keep the app private or invite only known viewers if logs are sensitive.

The dependency file is next to the entrypoint:

```text
dashboard/requirements.txt
```

## Security Model

- Uploaded logs are parsed as text only; the app never executes uploaded content.
- Uploaded logs are not written to disk.
- File type is restricted to `.log` and `.txt`.
- File size is capped by `MAX_UPLOAD_MB` and `.streamlit/config.toml`.
- Optional password gate is enabled with `DASHBOARD_PASSWORD` or Streamlit secret `dashboard_password`.
- Docker image runs as a non-root user.
- Compose runs with `read_only`, `no-new-privileges`, dropped Linux capabilities, and `/tmp` mounted as tmpfs.

Do not expose a public unauthenticated dashboard if logs contain strategy logic,
counterparty signals, or backtest output from current rounds.
