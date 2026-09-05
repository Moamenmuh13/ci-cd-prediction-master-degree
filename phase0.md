# Phase 0: Collect Real Dataset from GitHub Actions API

ARCHIVE the existing project first:

1. Rename current `cicd-failure-prediction/` to `cicd-failure-prediction-synthetic-archive/`
2. Create fresh `cicd-failure-prediction/` with same directory structure
3. Copy these REUSABLE files from the archive to the new project (we keep the code, replace the data):
   - src/visualization.py (the ThesisPlotter — keep as is)
   - src/utils.py (logging helpers — keep as is)
   - src/hybrid_pipeline.py (will need minor edits later for new schema)
   - requirements.txt
   - .gitignore
- the api token <REDACTED — set GITHUB_TOKEN env var>

## Task: Build a real CI/CD dataset from GitHub Actions API

We need ~10,000 workflow runs (build outcomes) with:
- Pass/fail status
- Commit messages
- Lines/files changed
- Repository metadata
- Author info
- Timestamps

### Target repositories to collect from

Curate a list of 15-20 active open-source repositories with heavy CI/CD activity.
Pick a mix of languages and project sizes for diversity. Examples (the script
should use these and add more as needed to reach the target row count):

- facebook/react
- microsoft/vscode
- tensorflow/tensorflow
- pytorch/pytorch
- vuejs/vue
- nodejs/node
- rust-lang/rust
- kubernetes/kubernetes
- ansible/ansible
- elastic/elasticsearch
- nestjs/nest
- expressjs/express
- pandas-dev/pandas
- scikit-learn/scikit-learn
- huggingface/transformers
- prisma/prisma
- vercel/next.js
- sveltejs/svelte
- ruby/ruby
- python/cpython

### Implementation

Create `src/collect_github_data.py`:

```python
"""
Collect real CI/CD workflow run data from GitHub Actions API.

For each target repository:
1. Fetch the most recent N workflow runs via GET /repos/{owner}/{repo}/actions/runs
2. For each workflow run, fetch the associated commit details via GET /repos/{owner}/{repo}/commits/{sha}
3. Extract: workflow status, commit message, lines added/deleted, files changed,
   author, timestamps, repository metadata
4. Append to a master CSV
"""
```

The script must:
- Accept GitHub token via environment variable `GITHUB_TOKEN`
- Use proper rate-limit handling (sleep when approaching limit)
- Use `requests` with session for connection pooling
- Save progress incrementally (so if it crashes, we don't lose all data)
- Use `tqdm` for progress bars
- Log everything to console + file

### Specific GitHub API endpoints to use

1. List workflow runs:
   GET /repos/{owner}/{repo}/actions/runs?per_page=100&page={N}
   Returns: status, conclusion, run_started_at, run_attempt, event, head_branch, head_commit, etc.

2. Get commit details:
   GET /repos/{owner}/{repo}/commits/{sha}
   Returns: commit.message, commit.author, stats.additions, stats.deletions,
            stats.total, files[].filename, files[].status, files[].changes

### Columns to extract per row

| Column | Source | Description |
|--------|--------|-------------|
| run_id | runs API | unique workflow run ID |
| repository | input | "owner/repo" |
| workflow_name | runs API | name of the workflow file |
| event | runs API | push / pull_request / schedule |
| branch | runs API | head_branch |
| conclusion | runs API | success / failure / cancelled / skipped |
| status | runs API | completed / in_progress / queued |
| created_at | runs API | timestamp |
| updated_at | runs API | timestamp |
| run_duration_sec | computed | (updated_at - created_at) |
| run_attempt | runs API | 1, 2, 3 (re-runs) |
| commit_sha | runs API | head_sha |
| commit_message | commits API | full commit message |
| commit_author | commits API | author.login |
| author_association | commits API | OWNER/MEMBER/CONTRIBUTOR |
| lines_added | commits API | stats.additions |
| lines_deleted | commits API | stats.deletions |
| total_changes | commits API | stats.total |
| files_changed | commits API | len(files) |
| commit_date | commits API | commit.author.date |

### Filtering rules

- Only keep runs where `status == "completed"` (skip in-progress)
- Only keep conclusions in {"success", "failure"} — drop cancelled, skipped, neutral
- Drop rows where commit_message is null or empty
- Drop rows where commit_message length < 5 characters
- Cap commit_message length at 1000 characters (truncate longer ones)

### Target distribution

- Aim for ~10,000 total rows across all repos
- Stop fetching from a repo once we have 500-1000 rows from it
- Move to the next repo when current repo is exhausted or capped
- Keep going until we hit ~10,000 OR all repos exhausted

### Output

Save to: `data/raw/github_actions_real.csv`

Print summary at the end:
- Total rows collected
- Distribution per repository
- Distribution of conclusion (success/failure ratio)
- Date range of collected data
- Average commit message length
- Top 5 most common authors

### Important constraints

- Set `GITHUB_TOKEN` as environment variable, never hardcode
- Respect GitHub rate limits: pause when X-RateLimit-Remaining < 100
- Use exponential backoff on 502/503 errors
- Set a global timeout of 4 hours (just in case)
- All code in English with type hints

### Setup instructions

Print at the top of the script comments how to run it:

```bash
export GITHUB_TOKEN="your_token_here"
python src/collect_github_data.py
```

### After collection

Run a quick EDA on the new dataset:

1. Load data/raw/github_actions_real.csv
2. Print shape, dtypes, missing values
3. Plot the success/failure distribution (this is the target!)
4. Plot the distribution of commit message lengths
5. Plot rows per repository (top 20)
6. Save these as fig_eda_01.png, fig_eda_02.png, fig_eda_03.png to figures/

This is just a sanity check — full EDA pipeline will come in Phase 1.

### Deliverables to me after script completes

1. The final row count
2. Success/failure ratio (very important — if it's 95/5 we have an imbalance problem)
3. List of top 5 repos by row count
4. 5 sample commit messages (random rows)
5. Average lines_added and files_changed (to confirm we have real signal)
6. The 3 EDA charts
7. Any errors or warnings during collection