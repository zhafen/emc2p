"""Regression coverage for scripts/check-no-downstream-mentions.sh
(.githooks/pre-commit's first check).

Runs the real script as a subprocess against a throwaway git repo, same
pattern as test_check_pr_size_hook.py. Scoped to the staged diff, not
the whole tree -- see the script's own comments for why.
"""

import subprocess
from pathlib import Path

import pytest

HOOK = Path(__file__).resolve().parents[1] / "scripts" / "check-no-downstream-mentions.sh"


def _run_hook(cwd):
    return subprocess.run(
        ["bash", str(HOOK), "--staged"],
        cwd=cwd,
        capture_output=True,
        text=True,
        check=False,
    )


def _git(*args, cwd):
    subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True)


@pytest.fixture
def git_repo(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    _git("init", "-q", "-b", "main", cwd=repo)
    _git("config", "user.email", "test@example.com", cwd=repo)
    _git("config", "user.name", "Test", cwd=repo)
    (repo / "docs" / "manifest").mkdir(parents=True)
    (repo / "docs" / "manifest" / "history.yaml").write_text("history: []\n")
    (repo / "scripts").mkdir()
    (repo / "scripts" / "check-no-downstream-mentions.sh").write_text("#!/bin/bash\n")
    _git("add", "-A", cwd=repo)
    _git("commit", "-q", "-m", "init", cwd=repo)
    return repo


class TestFlagged:
    def test_flags_a_newly_staged_mention(self, git_repo):
        (git_repo / "README.md").write_text("mentions story-simulator here\n")
        _git("add", "README.md", cwd=git_repo)

        result = _run_hook(git_repo)

        assert result.returncode == 1
        assert "README.md" in result.stdout
        assert "story-simulator" in result.stdout

    def test_case_and_separator_insensitive(self, git_repo):
        (git_repo / "README.md").write_text("Story_Sim is the downstream project\n")
        _git("add", "README.md", cwd=git_repo)

        result = _run_hook(git_repo)

        assert result.returncode == 1


class TestExempt:
    def test_history_yaml_is_exempt(self, git_repo):
        (git_repo / "docs" / "manifest" / "history.yaml").write_text(
            "history: mentions story-simulator here\n"
        )
        _git("add", "docs/manifest/history.yaml", cwd=git_repo)

        result = _run_hook(git_repo)

        assert result.returncode == 0

    def test_self_path_is_exempt(self, git_repo):
        (git_repo / "scripts" / "check-no-downstream-mentions.sh").write_text(
            "#!/bin/bash\n# story-simulator\n"
        )
        _git("add", "scripts/check-no-downstream-mentions.sh", cwd=git_repo)

        result = _run_hook(git_repo)

        assert result.returncode == 0

    def test_mentioning_iacs_alone_is_fine(self, git_repo):
        """emc2p was developed as part of iacs -- only naming a separate
        downstream project (story-simulator) is what this guards against."""
        (git_repo / "README.md").write_text("originally developed as part of iacs\n")
        _git("add", "README.md", cwd=git_repo)

        result = _run_hook(git_repo)

        assert result.returncode == 0


class TestClean:
    def test_unrelated_staged_change_passes(self, git_repo):
        (git_repo / "README.md").write_text("nothing downstream-specific here\n")
        _git("add", "README.md", cwd=git_repo)

        result = _run_hook(git_repo)

        assert result.returncode == 0
        assert result.stdout == ""

    def test_pre_existing_mention_elsewhere_does_not_block_an_unrelated_commit(self, git_repo):
        """The whole point of scoping to the diff, not the tree: a
        pre-existing violation somewhere else must not block a commit
        that never touches it."""
        (git_repo / "OLD.md").write_text("mentions story-simulator, committed before this hook\n")
        _git("add", "OLD.md", cwd=git_repo)
        _git("commit", "-q", "-m", "pre-existing violation", cwd=git_repo)

        (git_repo / "README.md").write_text("totally unrelated change\n")
        _git("add", "README.md", cwd=git_repo)

        result = _run_hook(git_repo)

        assert result.returncode == 0
