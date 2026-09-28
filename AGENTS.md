# AGENTS.md — ai-where server (xiaozhi-esp32-server fork)

## TDD is mandatory here — no exceptions

Every production code change (fix, feature, refactor) ships with a test that was
**seen to FAIL first** (RED), then pass (GREEN). This is the Iron Law:

```
NO PRODUCTION CODE WITHOUT A FAILING TEST FIRST
```

- **Bug fix** → write the regression test, revert the fix, watch it FAIL with the
  user's actual symptom, restore the fix, watch it pass.
- **New feature** → write the failing test, then implement minimally to pass.
- A test that never ran RED against the buggy code guards nothing — it is not
  evidence the fix works.

Load the `test-driven-development` skill for the full RED-GREEN-REFACTOR cycle and
the verify-RED recipe before writing code here.

## Repo facts

- This is the **server fork**, NOT the iOS client and NOT the outer `~/Projects/ai-where`
  repo (that outer repo holds firmware/docs/VERSION and ignores this directory).
- Version = git tags (`vX.Y.Z`) + conventional commits (`fix:` / `feat:`) **and**
  `SERVER_VERSION` in `main/xiaozhi-server/config/logger.py` — keep both in sync on
  every bump (v0.9.6 drifted from tag v0.9.7; fixed at v0.9.8).
- Tests: `cd main/xiaozhi-server && source .venv/bin/activate && python -m pytest tests/ -q`.
  `tests/conftest.py` adds the project root to `sys.path`.
- Running service: launchd `com.ai-where.xiaozhi-server` (KeepAlive). Restart with
  `launchctl kickstart -k gui/$(id -u)/com.ai-where.xiaozhi-server`. Logs at
  `main/xiaozhi-server/tmp/server.log`.
