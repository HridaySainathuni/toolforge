# Plan: Fix, Improve, Push README to GitHub

## Context
ToolForge has accumulated uncommitted changes (config fix, UI overhaul, new API endpoints, favicon) and lacks a README. We need to commit everything cleanly and push to `origin/master` along with a comprehensive README.

## Steps

### 1. Update `.gitignore` + un-track binary databases
- Add `*.db`, `.coverage`, `htmlcov/`, `.pytest_cache/` to `.gitignore`
- `git rm --cached library/tool_library.db library/failures.db` (they're regenerated from seed data on startup)
- Commit: `chore: add *.db and test artifacts to .gitignore`

### 2. Commit feature changes
Stage: `config.py`, `web/app.py`, `web/templates/index.html`, `web/static/favicon.ico`
- config.py: explicit dotenv path, port 5001
- web/app.py: favicon route, tool source/delete/stop endpoints
- index.html: example chips, tool search, view source, copy button, stop button, collapsible iterations, elapsed time, two-click delete, clear chat fix
- favicon.ico: new static asset

Commit: `feat: improved web UI with new features and config fixes`

### 3. Create and commit README.md
Structure:
- Project name + tagline
- What It Does (autonomous tool acquisition concept)
- Architecture (agent loop, tool library, retriever, generator, failure store, librarian, sandbox, web UI)
- Quick Start (clone, .env, pip install, run, open browser)
- API Endpoints table (7 endpoints)
- Evaluation section
- Tech Stack
- Project Structure tree

Commit: `docs: add comprehensive README`

### 4. Push to GitHub
`git push origin master` — single-developer repo, no branch protection.

## Files to modify/create
- `.gitignore` — add patterns
- `README.md` — new file
- Stage existing changes: `config.py`, `web/app.py`, `web/templates/index.html`, `web/static/favicon.ico`

## Verification
- `git log --oneline -5` to confirm 3 new commits
- `git status` should be clean (except .db files now untracked)
- `gh repo view` or check GitHub to confirm push succeeded
- README renders correctly on GitHub
