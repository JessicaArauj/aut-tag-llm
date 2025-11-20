#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd -- "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${PROJECT_ROOT}/.venv"

echo "[aut-tests] Bootstrapping virtual environment..."
if [[ ! -d "${VENV_DIR}" ]]; then
  python -m venv "${VENV_DIR}"
fi

source "${VENV_DIR}/Scripts/activate"

echo "[aut-tests] Installing dependencies..."
python -m pip install -q -U pip >/dev/null
python -m pip install -q -r "${PROJECT_ROOT}/requirements.txt"

echo "[aut-tests] Running pytest suite..."
cd "${PROJECT_ROOT}"
mkdir -p "${PROJECT_ROOT}/output"
python -m pytest --json-report --json-report-file "${PROJECT_ROOT}/output/test_results.json" "$@"

echo "[aut-tests] Saving dashboard snapshot..."
python -m script.python.save_dashboard_snapshot || echo "[aut-tests] Snapshot generation failed."

DASHBOARD_ENTRY="${PROJECT_ROOT}/dashboards/test_metrics_dashboard.py"
if [[ -f "${DASHBOARD_ENTRY}" ]]; then
  echo "[aut-tests] Launching Streamlit dashboard (press Ctrl+C to stop)..."
  python -m streamlit run "${DASHBOARD_ENTRY}" --server.headless true
else
  echo "[aut-tests] Dashboard entry not found at ${DASHBOARD_ENTRY}; skipping Streamlit step."
  echo "           Create dashboards/test_metrics_dashboard.py to enable"
fi
