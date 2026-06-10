# Security Policy

## Scope
EverMate is a local-first desktop app. The main assets to protect are:
- the user's memory directory (chat logs, imported documents, index) —
  plain files under the per-user data directory;
- the promise that no data leaves the machine by default.

## Reporting a vulnerability
Open a GitHub issue with the label `security`, or contact the maintainer
privately if the issue exposes user data. Please include reproduction steps.
You should expect an acknowledgement within a week.

## Known design notes
- Memory files are stored **unencrypted**; rely on OS-level disk encryption
  (FileVault) for at-rest protection.
- Imported document text is fenced when injected into prompts, and the
  system prompt instructs the model not to follow instructions inside it —
  but prompt injection against local LLMs can never be fully ruled out.
  Treat imported documents from untrusted sources accordingly.
- `OLLAMA_URL` may be pointed at a remote host; traffic is plain HTTP
  unless you provide an HTTPS endpoint. The default is localhost.
- Benchmark scripts can use a cloud judge (`GOOGLE_API_KEY`); that path
  sends corpus excerpts to the cloud and must never be used on private data.
