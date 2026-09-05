"""Collect real CI/CD workflow-run data from the GitHub Actions API.

For each target repository:

1. Fetch the most recent workflow runs via
   ``GET /repos/{owner}/{repo}/actions/runs?status=completed``.
2. For each kept run, fetch commit details via
   ``GET /repos/{owner}/{repo}/commits/{sha}`` to enrich the row with
   ``commit_message``, ``lines_added``, ``lines_deleted`` and
   ``files_changed``.
3. Apply the Phase 0 filtering rules and append the resulting row to a
   master CSV.

Usage::

    export GITHUB_TOKEN="your_personal_access_token"
    python src/collect_github_data.py

The script writes ``data/raw/github_actions_real.csv`` and logs to both
stdout and ``logs/collect_github_data.log``. Progress is saved
incrementally — ``f.flush()`` after every row — so a crash never wastes
hours of API calls.
"""
from __future__ import annotations

import csv
import os
import random
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator

import requests
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils import RAW_DATA_DIR, ensure_dir, get_logger  # noqa: E402


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

GITHUB_API_BASE = "https://api.github.com"

TARGET_TOTAL_ROWS = 10_000
PER_REPO_CAP = 600
PER_PAGE = 100
MAX_PAGES_PER_REPO = 30

RATE_LIMIT_REMAINING_FLOOR = 100
RATE_LIMIT_BUFFER_SECONDS = 5
MAX_RETRIES_PER_REQUEST = 5
BACKOFF_BASE_SECONDS = 5
GLOBAL_TIMEOUT_SECONDS = 4 * 60 * 60  # 4 hours

REQUEST_TIMEOUT_SECONDS = 30


TARGET_REPOSITORIES: list[str] = [
    "facebook/react",
    "microsoft/vscode",
    "tensorflow/tensorflow",
    "pytorch/pytorch",
    "vuejs/vue",
    "nodejs/node",
    "rust-lang/rust",
    "kubernetes/kubernetes",
    "ansible/ansible",
    "elastic/elasticsearch",
    "nestjs/nest",
    "expressjs/express",
    "pandas-dev/pandas",
    "scikit-learn/scikit-learn",
    "huggingface/transformers",
    "prisma/prisma",
    "vercel/next.js",
    "sveltejs/svelte",
    "ruby/ruby",
    "python/cpython",
]


CSV_COLUMNS: list[str] = [
    "run_id",
    "repository",
    "workflow_name",
    "event",
    "branch",
    "conclusion",
    "status",
    "created_at",
    "updated_at",
    "run_duration_sec",
    "run_attempt",
    "commit_sha",
    "commit_message",
    "commit_author",
    "author_association",
    "lines_added",
    "lines_deleted",
    "total_changes",
    "files_changed",
    "commit_date",
]


_PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_CSV = RAW_DATA_DIR / "github_actions_real.csv"
LOG_FILE = _PROJECT_ROOT / "logs" / "collect_github_data.log"


_LOGGER = get_logger(__name__, log_file=LOG_FILE)


# --------------------------------------------------------------------------- #
# HTTP helpers
# --------------------------------------------------------------------------- #


def build_session(token: str) -> requests.Session:
    """Return a ``requests`` session configured for GitHub API access."""
    session = requests.Session()
    session.headers.update(
        {
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "ci-cd-failure-thesis-collector/0.1",
        }
    )
    return session


def _maybe_sleep_for_rate_limit(headers: dict[str, str]) -> None:
    try:
        remaining = int(headers.get("X-RateLimit-Remaining", "5000"))
        reset_at = int(headers.get("X-RateLimit-Reset", str(int(time.time()) + 60)))
    except (TypeError, ValueError):
        return
    if remaining < RATE_LIMIT_REMAINING_FLOOR:
        wait_seconds = max(
            reset_at - int(time.time()) + RATE_LIMIT_BUFFER_SECONDS,
            10,
        )
        _LOGGER.warning(
            "Approaching rate limit (remaining=%d). Sleeping %ds until reset.",
            remaining,
            wait_seconds,
        )
        time.sleep(wait_seconds)


def github_get(
    session: requests.Session,
    url: str,
    params: dict[str, Any] | None = None,
    *,
    max_retries: int = MAX_RETRIES_PER_REQUEST,
) -> Any | None:
    """GET with retries, exponential backoff, and rate-limit handling.

    Returns the decoded JSON body on 200. Returns ``None`` for 404/410 and
    other unrecoverable client errors.
    """
    for attempt in range(max_retries):
        try:
            response = session.get(
                url, params=params, timeout=REQUEST_TIMEOUT_SECONDS
            )
        except requests.RequestException as exc:
            wait = BACKOFF_BASE_SECONDS * (2**attempt) + random.uniform(0, 1)
            _LOGGER.warning(
                "Network error %r on %s; backing off %.1fs", exc, url, wait
            )
            time.sleep(wait)
            continue

        status = response.status_code

        if status == 200:
            _maybe_sleep_for_rate_limit(response.headers)
            try:
                return response.json()
            except ValueError:
                return None

        if status in (404, 410, 422):
            return None

        if status == 401:
            _LOGGER.error("401 Unauthorized — check GITHUB_TOKEN env var.")
            return None

        if status == 403:
            remaining_str = response.headers.get("X-RateLimit-Remaining", "1")
            try:
                remaining_int = int(remaining_str)
            except ValueError:
                remaining_int = 1
            if remaining_int == 0 or "rate limit" in response.text.lower():
                _maybe_sleep_for_rate_limit(response.headers)
                continue
            _LOGGER.warning(
                "403 Forbidden on %s (non-rate-limit): %s",
                url,
                response.text[:200],
            )
            return None

        if status in (500, 502, 503, 504):
            wait = BACKOFF_BASE_SECONDS * (2**attempt) + random.uniform(0, 1)
            _LOGGER.warning(
                "HTTP %d from %s; backing off %.1fs", status, url, wait
            )
            time.sleep(wait)
            continue

        _LOGGER.warning(
            "Unexpected HTTP %d on %s — body=%s",
            status,
            url,
            response.text[:200],
        )
        return None

    _LOGGER.error("Gave up on %s after %d retries.", url, max_retries)
    return None


# --------------------------------------------------------------------------- #
# Domain-level helpers
# --------------------------------------------------------------------------- #


def iter_workflow_runs(
    session: requests.Session, owner: str, repo: str, max_pages: int = MAX_PAGES_PER_REPO
) -> Iterator[dict]:
    """Yield workflow runs (newest first) for ``owner/repo``."""
    page = 1
    while page <= max_pages:
        url = f"{GITHUB_API_BASE}/repos/{owner}/{repo}/actions/runs"
        payload = github_get(
            session,
            url,
            params={
                "per_page": PER_PAGE,
                "page": page,
                "status": "completed",
            },
        )
        if not isinstance(payload, dict):
            return
        runs = payload.get("workflow_runs") or []
        if not runs:
            return
        for run in runs:
            yield run
        if len(runs) < PER_PAGE:
            return
        page += 1


def fetch_commit(
    session: requests.Session, owner: str, repo: str, sha: str
) -> dict | None:
    url = f"{GITHUB_API_BASE}/repos/{owner}/{repo}/commits/{sha}"
    payload = github_get(session, url)
    return payload if isinstance(payload, dict) else None


def _parse_iso(timestamp: str | None) -> datetime | None:
    if not timestamp:
        return None
    try:
        return datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
    except ValueError:
        return None


def _compute_duration(created_at: str | None, updated_at: str | None) -> float | None:
    ca = _parse_iso(created_at)
    ua = _parse_iso(updated_at)
    if ca is None or ua is None:
        return None
    return (ua - ca).total_seconds()


def normalise_row(
    run: dict, commit: dict, repository: str
) -> dict | None:
    """Apply Phase 0 filtering rules and return a flat CSV row dict."""
    status = run.get("status")
    conclusion = run.get("conclusion")
    if status != "completed":
        return None
    if conclusion not in {"success", "failure"}:
        return None

    commit_inner = commit.get("commit") or {}
    commit_message = (commit_inner.get("message") or "").strip()
    if not commit_message or len(commit_message) < 5:
        return None
    if len(commit_message) > 1000:
        commit_message = commit_message[:1000]

    stats = commit.get("stats") or {}
    files = commit.get("files") or []
    author = commit.get("author")
    author_login = author.get("login") if isinstance(author, dict) else None
    commit_author_meta = commit_inner.get("author") or {}

    return {
        "run_id": run.get("id"),
        "repository": repository,
        "workflow_name": run.get("name"),
        "event": run.get("event"),
        "branch": run.get("head_branch"),
        "conclusion": conclusion,
        "status": status,
        "created_at": run.get("created_at"),
        "updated_at": run.get("updated_at"),
        "run_duration_sec": _compute_duration(
            run.get("created_at"), run.get("updated_at")
        ),
        "run_attempt": run.get("run_attempt"),
        "commit_sha": run.get("head_sha"),
        "commit_message": commit_message,
        "commit_author": author_login,
        "author_association": commit.get("author_association"),
        "lines_added": stats.get("additions"),
        "lines_deleted": stats.get("deletions"),
        "total_changes": stats.get("total"),
        "files_changed": len(files),
        "commit_date": commit_author_meta.get("date"),
    }


def load_existing_run_ids(csv_path: Path) -> set[int]:
    """Return ``run_id`` values already present in ``csv_path`` (for resume)."""
    if not csv_path.exists():
        return set()
    seen: set[int] = set()
    try:
        with csv_path.open("r", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                raw = row.get("run_id")
                if raw is None:
                    continue
                try:
                    seen.add(int(raw))
                except ValueError:
                    continue
    except Exception as exc:  # pragma: no cover — defensive
        _LOGGER.warning("Could not read existing CSV for resume: %s", exc)
    return seen


# --------------------------------------------------------------------------- #
# Collection driver
# --------------------------------------------------------------------------- #


def collect() -> int:
    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        print(
            "ERROR: GITHUB_TOKEN environment variable is not set.",
            file=sys.stderr,
        )
        sys.exit(2)

    ensure_dir(OUTPUT_CSV.parent)
    ensure_dir(LOG_FILE.parent)

    session = build_session(token)
    seen_run_ids = load_existing_run_ids(OUTPUT_CSV)
    _LOGGER.info(
        "Resuming with %d previously collected run_ids.", len(seen_run_ids)
    )
    total_rows = len(seen_run_ids)
    write_header = not OUTPUT_CSV.exists()

    start_monotonic = time.monotonic()

    with OUTPUT_CSV.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=CSV_COLUMNS, extrasaction="ignore"
        )
        if write_header:
            writer.writeheader()
            handle.flush()

        repo_pbar = tqdm(
            TARGET_REPOSITORIES, desc="Repositories", unit="repo", position=0
        )
        for repo_slug in repo_pbar:
            if total_rows >= TARGET_TOTAL_ROWS:
                _LOGGER.info(
                    "Target total reached (%d). Stopping further collection.",
                    TARGET_TOTAL_ROWS,
                )
                break

            elapsed = time.monotonic() - start_monotonic
            if elapsed > GLOBAL_TIMEOUT_SECONDS:
                _LOGGER.warning(
                    "Global timeout reached after %.0fs. Stopping.", elapsed
                )
                break

            try:
                owner, repo = repo_slug.split("/")
            except ValueError:
                _LOGGER.warning("Skipping malformed repo slug: %s", repo_slug)
                continue

            repo_pbar.set_postfix_str(
                f"{repo_slug} (total {total_rows})"
            )
            _LOGGER.info("=== %s — starting ===", repo_slug)

            kept_for_repo = 0
            scanned_for_repo = 0

            row_pbar = tqdm(
                total=PER_REPO_CAP,
                desc=f"  {repo_slug}",
                unit="row",
                leave=False,
                position=1,
            )

            try:
                for run in iter_workflow_runs(session, owner, repo):
                    if kept_for_repo >= PER_REPO_CAP:
                        break
                    if total_rows >= TARGET_TOTAL_ROWS:
                        break
                    if (
                        time.monotonic() - start_monotonic
                        > GLOBAL_TIMEOUT_SECONDS
                    ):
                        break

                    scanned_for_repo += 1
                    run_id = run.get("id")
                    if run_id in seen_run_ids:
                        continue
                    if run.get("status") != "completed":
                        continue
                    if run.get("conclusion") not in {"success", "failure"}:
                        continue
                    sha = run.get("head_sha")
                    if not sha:
                        continue

                    commit = fetch_commit(session, owner, repo, sha)
                    if commit is None:
                        continue

                    row = normalise_row(run, commit, repo_slug)
                    if row is None:
                        continue

                    writer.writerow(row)
                    handle.flush()
                    seen_run_ids.add(run_id)
                    kept_for_repo += 1
                    total_rows += 1
                    row_pbar.update(1)
                    repo_pbar.set_postfix_str(
                        f"{repo_slug} (total {total_rows})"
                    )
            finally:
                row_pbar.close()

            _LOGGER.info(
                "%s: kept %d / scanned %d (running total: %d)",
                repo_slug,
                kept_for_repo,
                scanned_for_repo,
                total_rows,
            )

    elapsed_total = time.monotonic() - start_monotonic
    _LOGGER.info(
        "Collection finished in %.1fs (%.2f minutes). Total rows: %d",
        elapsed_total,
        elapsed_total / 60.0,
        total_rows,
    )
    return total_rows


# --------------------------------------------------------------------------- #
# Summary
# --------------------------------------------------------------------------- #


def print_summary() -> None:
    if not OUTPUT_CSV.exists():
        print("No CSV produced — nothing to summarise.")
        return

    import pandas as pd

    df = pd.read_csv(OUTPUT_CSV)

    print()
    print("=" * 72)
    print("Collection summary")
    print("=" * 72)
    print(f"Total rows         : {len(df):,}")
    print(f"Distinct repos     : {df['repository'].nunique()}")

    if "created_at" in df.columns and not df["created_at"].isna().all():
        print(
            "Date range         : "
            f"{df['created_at'].min()}  →  {df['created_at'].max()}"
        )

    if "commit_message" in df.columns:
        avg_len = df["commit_message"].dropna().astype(str).str.len().mean()
        print(f"Avg commit msg len : {avg_len:.1f} chars")

    if "conclusion" in df.columns:
        print("\nDistribution by conclusion:")
        print(df["conclusion"].value_counts(dropna=False).to_string())

    if "repository" in df.columns:
        print("\nRows per repository:")
        print(df["repository"].value_counts().to_string())

    if "commit_author" in df.columns:
        print("\nTop 5 commit authors:")
        print(df["commit_author"].value_counts().head(5).to_string())


def main() -> None:
    collect()
    print_summary()


if __name__ == "__main__":
    main()
