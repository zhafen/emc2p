"""Unit tests for emc2p.testing.agent_session's create_agent_session/AgentSession.

Only constructor dispatch is under test here -- neither __enter__ (which
would spawn a real subprocess/MCP server) is exercised. HeadlessSession's
own timeout/kill mechanics are already covered by test_headless_session.py;
McpClientSession's own connect/serve loop by test_mcp_client_session.py.
"""

import pytest

from emc2p.testing.agent_session import AgentSession, create_agent_session
from emc2p.testing.headless_session import HeadlessSession
from emc2p.testing.mcp_client_session import McpClientSession


class TestCreateAgentSession:
    def test_claude_driver_returns_a_headless_session_with_claude_provider(self, tmp_path):
        session = create_agent_session(
            "claude", mcp_config=tmp_path / ".mcp.json", allowed_tools=["mcp__foo__*"],
            cwd=tmp_path, model="claude-sonnet-5",
        )
        assert isinstance(session, HeadlessSession)
        assert session.provider.name == "claude"

    def test_copilot_driver_returns_a_headless_session_with_copilot_provider(self, tmp_path):
        session = create_agent_session(
            "copilot", mcp_config=tmp_path / ".mcp.json", allowed_tools=["mcp__foo__*"],
            cwd=tmp_path, model="gpt-4o",
        )
        assert isinstance(session, HeadlessSession)
        assert session.provider.name == "copilot"

    def test_copilot_driver_passes_provider_options_through(self, tmp_path):
        session = create_agent_session(
            "copilot", mcp_config=tmp_path / ".mcp.json", allowed_tools=["mcp__foo__*"],
            cwd=tmp_path, model="gpt-4o", provider_options={"some_flag": True},
        )
        assert session.provider_options == {"some_flag": True}

    def test_mcp_client_driver_returns_an_mcp_client_session(self, tmp_path):
        session = create_agent_session(
            "mcp_client", mcp_config=tmp_path / ".mcp.json", allowed_tools=["mcp__foo__*"],
            cwd=tmp_path, model="deepseek/deepseek-chat",
        )
        assert isinstance(session, McpClientSession)

    def test_shared_arguments_are_forwarded_regardless_of_driver(self, tmp_path):
        session = create_agent_session(
            "mcp_client", mcp_config=tmp_path / ".mcp.json", allowed_tools=["mcp__foo__*"],
            cwd=tmp_path, model="deepseek/deepseek-chat",
            extra_env={"SOME_VAR": "1"}, turn_timeout=10, session_timeout=20,
        )
        assert session.extra_env == {"SOME_VAR": "1"}
        assert session.turn_timeout == 10
        assert session.session_timeout == 20

    def test_unknown_driver_raises(self, tmp_path):
        with pytest.raises(ValueError, match="ollama"):
            create_agent_session(
                "ollama", mcp_config=tmp_path / ".mcp.json", allowed_tools=[],
                cwd=tmp_path, model="whatever",
            )


class TestAgentSessionProtocol:
    def test_headless_session_satisfies_the_protocol(self, tmp_path):
        session = HeadlessSession(
            mcp_config=tmp_path / ".mcp.json", allowed_tools=[], cwd=tmp_path, model="claude-sonnet-5",
        )
        assert isinstance(session, AgentSession)

    def test_mcp_client_session_satisfies_the_protocol(self, tmp_path):
        session = McpClientSession(
            mcp_config=tmp_path / ".mcp.json", allowed_tools=[], cwd=tmp_path, model="deepseek/deepseek-chat",
        )
        assert isinstance(session, AgentSession)
