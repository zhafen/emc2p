"""Infrastructure for live end-to-end tests that drive a real MCP server
through a real MCP client, plus generic helpers for checking what such a
client actually wrote to an emc2p registry.

Nothing here knows about any downstream project's own domain (entities,
component schemas, MCP tool names) -- see `headless_session.HeadlessSession`
and `registry_checks` for what's generic versus what a downstream project
still has to supply itself.
"""
