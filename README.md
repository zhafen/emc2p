# emc2p

emc2p is the generic Entity-Component System (ECS) engine originally
developed as part of [`iacs`](https://github.com/zhafen/iacs): a
`Registry`/`Registrar` for storing and querying component data, the ETL
pipeline for loading entity-centered YAML/Python manifests into it and
exporting it back out, and the Hamilton-based dataflow execution system
those pipelines run on.

It has no notion of what the entities and components represent -- domain
concepts like requirements, cost/impact scoring, or architecture
diagramming live in a separate downstream project (`iacs`) that depends
on emc2p and adds its own component definitions and dataflows on top.

## Development

- Use `uv` for Python package management.
- Run tests with `uv run pytest`.

## Headless MCP test sessions (Claude + Copilot)

`emc2p.testing.headless_session.HeadlessSession` now supports provider-based
headless clients:

- Default behavior is unchanged (`provider="claude"`).
- Opt in to Copilot with `provider="copilot"`.
- Pass `provider_options` for provider-specific CLI/flag overrides.
- Tool isolation is capability-based: providers that support hard tool stripping
  do so directly; others receive a strict prompt preamble fallback listing the
  allowed MCP tools.

Migration path for downstream users:

1. Keep current tests unchanged for Claude compatibility.
2. Add a second test mode with `provider="copilot"` in your harness.
3. Start with defaults, then add `provider_options` if your Copilot CLI uses
   different argument names.
4. Compare behavior in failure handling, usage metrics, and tool-isolation
   semantics between providers before making Copilot mode required in CI.
