# Frank Reader — notes for Claude

Local web tool converting documents (PDF/DOCX/image/URL/text) into Ilya
Frank-method interlinear reading text via a local LLM. Design docs: SPEC.md,
SPEC-IMPL.md. Everything installs via uv only (`uv sync`) — no Homebrew/Docker.

- Run: `uv run frank-reader` (serves http://127.0.0.1:8200; port 8000 is taken
  on this machine). Data lives in `~/.frank-reader/`.
- Tests: `uv run pytest` (no live LLM needed; one integration test behind
  `FRANK_INTEGRATION=1`).
- LLM: LocalAI at http://127.0.0.1:1240/v1, model gemma-4-e4b-it-qat-q4_0 — a
  small model; its quirks (reasoning_effort, reasoning_content fallback, JSON
  fences, prompt echo) are handled in pipeline/llm_client.py and prompts.py.
  Any prompt text change must bump PROMPT_VERSION (invalidates the LLM cache).
- Public repo, entirely in English (code, comments, docs, UI).
- Commits are signed via 1Password; if it is locked, commit fails with "agent
  returned an error" — ask the user to unlock, never disable signing.

## Delegation

The main session is the orchestrator: it plans, reviews, and answers questions.
Delegate implementation to the `worker` agent using these rules:

- **Do it yourself (no delegation):** editing 1–2 files in a precisely known
  location, answering a question, reading a single file. Spawning an agent
  here is pure overhead.
- **Send a follow-up task to a live worker (SendMessage):** the next task
  touches the same code the worker just worked on, and no more than a couple
  of minutes have passed.
- **Spawn a new worker:** the topic/subsystem changed, the previous agent
  already completed a large task (its context is bloated), or the tasks are
  independent — in that case spawn several new workers in parallel.

After delegating, always review the resulting diff yourself.
