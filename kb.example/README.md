# Example knowledge base (fake)

Everything here is made up. It shows the layout `kb/` must follow, and the tests load it.

To run the agent as yourself, copy this folder to `kb/` and rewrite it. `kb/` is gitignored, so
your real details never reach the repo. See `src/voice_agent/core/prompt.py` for the rules:

- Required files: profile, status, experience, skills, faq, boundaries (`.md`).
- `projects/*.md` are optional, one per project, each starting with a `# Title` line.
- Front matter: `updated: YYYY-MM-DD` in status.md; `featured: yes|no` and `link:` in projects.
- Write in the first person, the way you'd say it on a call.
