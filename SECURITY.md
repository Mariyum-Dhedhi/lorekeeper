# Security Policy

## Reporting a vulnerability
Please report vulnerabilities **privately** using GitHub's [private vulnerability reporting](../../security/advisories/new). Do not open a public issue. I'll aim to respond within a week.

## Secrets
- API keys live only in a local `.env` file, which is gitignored. Never commit it.
- `.env.example` shows the variable names with empty values.
- The project needs no paid keys. Ollama runs locally and needs no key. A Groq key is optional; if you use one, keep it in `.env`.
- GitHub secret scanning and push protection are enabled on this repository.

## Data and network
- By default, the only network calls are:
  - to your local Ollama server (`localhost:11434`)
  - to the Groq API, if you enable it
  - a one-time download of the embedding model from Hugging Face
- Chroma's anonymous telemetry and Streamlit's usage statistics are turned off.
- The Streamlit app binds to `localhost` only (`.streamlit/config.toml`), so other devices on your network can't reach it.
- Model output is rendered as plain Markdown. Raw HTML from the model is never rendered.

## Known advisories
`chromadb` has open advisories (PYSEC-2026-311, -3813, -3814, -3815), none of which has a patched release yet. They affect Chroma's **HTTP server** and its multi-tenant authorization. This project runs Chroma **embedded in-process** and never starts that server, so they can't be reached here. Do not expose a `chroma run` server based on this setup. Dependabot will propose an upgrade once a fix ships.

## Supported versions
Only the latest commit on `main` is supported.
