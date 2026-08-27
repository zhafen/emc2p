"""Regression coverage for `emc2p.testing.responder_fixture.make_responder_fixture`."""

import pytest

from emc2p.testing.responder_fixture import make_responder_fixture

_ENV_VAR = "EMC2P_TEST_RESPONDER_FIXTURE_ENV_VAR"

responder = make_responder_fixture(_ENV_VAR, "the_default")
custom_named_responder = make_responder_fixture(_ENV_VAR, "the_default", name="custom_named_responder")


class TestMakeResponderFixture:
    def test_defaults_when_no_explicit_param_and_no_env_var(self, responder, monkeypatch):
        monkeypatch.delenv(_ENV_VAR, raising=False)
        assert responder == "the_default"

    def test_reads_the_env_var_when_no_explicit_param(self, request, monkeypatch):
        # responder must be fetched *after* setenv, not requested as a normal
        # fixture parameter -- pytest resolves directly-requested fixtures
        # before the test body runs, so setenv here would come too late.
        monkeypatch.setenv(_ENV_VAR, "from_env")
        assert request.getfixturevalue("responder") == "from_env"

    @pytest.mark.parametrize("responder", ["explicit"], indirect=True)
    def test_explicit_param_wins_over_the_env_var(self, responder, monkeypatch):
        monkeypatch.setenv(_ENV_VAR, "from_env")
        assert responder == "explicit"

    def test_fixture_is_registered_under_the_given_name(self, request, monkeypatch):
        # The registered fixture name comes from `name=`, not the wrapped
        # function's own __name__ (both factories above wrap a function
        # literally named `_responder`) -- requestable as
        # "custom_named_responder" proves the override took effect.
        monkeypatch.delenv(_ENV_VAR, raising=False)
        assert request.getfixturevalue("custom_named_responder") == "the_default"
