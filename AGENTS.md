# AGENTS.md

## Agent skills

### Issue tracker

Issues and specs live in GitHub Issues for `pupeno/ventris`. Before working with issues or specs, read `docs/agents/issue-tracker.md`.

### Triage labels

The five canonical triage roles use their default label names. Before triaging or applying triage labels, read `docs/agents/triage-labels.md`.

### Domain docs

This repo is single-context: a root `GLOSSARY.md` and `docs/adr/`. Before exploring the codebase, read `docs/agents/domain.md`.

## Other

Read README.md to orient yourself, learn what checks to run, how to activate the environment, run commands, etc.

Order functions and methods outside in: high-level entry points and orchestration first, followed by the helpers they call, then lower-level details. A linear read should establish the big picture before the implementation details.

Before declaring a task finished, and before making commits, make sure all formatting and checks pass. Run tests only when necessary (comment changes don't affect tests))

If you are about to put files in /tmp, put them in this repo's tmp instead, so it's easy to read them.
