"""Tests for the export_manifest dataflow."""

import yaml

from emc2p.registrar import Registrar


class TestExportManifestSplitsByFile:

    def test_export_manifest_splits_by_file(self, tmp_path):
        """Export should produce one output file per original source file."""
        input_dir = "examples/networks"
        output_dir = str(tmp_path)

        a = Registrar.from_manifest(input_dir)
        a.execute("etl.export_manifest", output_dir=output_dir)

        # One file per original source (builtins excluded)
        output_files = sorted(f.name for f in tmp_path.glob("*.yaml"))
        assert output_files == ["net_AB.yaml", "net_ABCD.yaml"]

        # Entities belong in the correct output file
        with open(tmp_path / "net_AB.yaml") as f:
            ab_data = yaml.safe_load(f)
        with open(tmp_path / "net_ABCD.yaml") as f:
            abcd_data = yaml.safe_load(f)

        assert "subnet_AB" in ab_data
        assert "net_ABCD" in abcd_data
        # No cross-contamination
        assert "subnet_AB" not in abcd_data
        assert "net_ABCD" not in ab_data


class TestPythonSourcedEntitiesExcludedFromExport:
    """Docstring-derived entities have no standalone YAML file to
    round-trip into.

    Including "Components" YAML sections -- they must not appear in
    exported manifest output.
    """

    def test_python_entities_not_written_to_output_dir(self, tmp_path):
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        (input_dir / "mod.py").write_text(
            'def my_function():\n'
            '    """Docstring here.\n'
            '\n'
            '    Components\n'
            '    ----------\n'
            '    - todo: We need to implement this.\n'
            '    """\n'
        )
        (input_dir / "req.yaml").write_text(
            "req_a:\n- description: Requirement A\n- requirement\n"
        )
        output_dir = tmp_path / "output"

        a = Registrar.from_manifest(str(input_dir))
        a.execute("etl.export_manifest", output_dir=str(output_dir))

        output_files = sorted(f.name for f in output_dir.glob("*"))
        assert output_files == ["req.yaml"]

        with open(output_dir / "req.yaml") as f:
            data = yaml.safe_load(f)
        assert "req_a" in data
        assert not any("my_function" in key for key in data)


class TestExportManifestCollapsesTimeDimensionedComponents:
    """A component type flagged time_dimension should export only its
    current row per entity, not its full accumulated history.

    A manifest export is a snapshot of current state, not a change log.
    """

    def test_repeated_description_writes_export_only_the_latest(self, tmp_path):
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        (input_dir / "thing.yaml").write_text("thing:\n- description: first\n")
        output_dir = tmp_path / "output"

        a = Registrar.from_manifest(str(input_dir))
        a.update(
            yaml_strings={"turn2": "thing:\n- same_as: thing\n- description: second\n"},
            time=2,
        )
        a.execute("etl.export_manifest", output_dir=str(output_dir))

        with open(output_dir / "thing.yaml") as f:
            text = f.read()
        assert "second" in text
        assert "first" not in text

    def test_non_time_dimensioned_component_still_exports_full_history(self, tmp_path):
        """A component type with no time_dimension field is unaffected -- every
        recorded instance is still exported, not just one.
        """
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        (input_dir / "thing.yaml").write_text("thing:\n- todo: first todo\n")
        output_dir = tmp_path / "output"

        a = Registrar.from_manifest(str(input_dir))
        a.update(yaml_strings={"turn2": "thing:\n- same_as: thing\n- todo: second todo\n"})
        a.execute("etl.export_manifest", output_dir=str(output_dir))

        with open(output_dir / "thing.yaml") as f:
            text = f.read()
        assert "first todo" in text
        assert "second todo" in text

    def test_collapse_time_dimension_false_exports_full_history(self, tmp_path):
        """Passing collapse_time_dimension=False opts out of collapsing
        entirely, even for a time_dimension'd component type.

        E.g. for a full-history export kept alongside a collapsed
        current-state one.
        """
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        (input_dir / "thing.yaml").write_text("thing:\n- description: first\n")
        output_dir = tmp_path / "output"

        a = Registrar.from_manifest(str(input_dir))
        a.update(
            yaml_strings={"turn2": "thing:\n- same_as: thing\n- description: second\n"},
            time=2,
        )
        a.execute("etl.export_manifest", output_dir=str(output_dir), collapse_time_dimension=False)

        with open(output_dir / "thing.yaml") as f:
            text = f.read()
        assert "first" in text
        assert "second" in text

    def test_export_manifest_registrar_method_accepts_collapse_time_dimension(self, tmp_path):
        """Registrar.export_manifest itself exposes the flag, not just the raw dataflow."""
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        (input_dir / "thing.yaml").write_text("thing:\n- description: first\n")
        output_dir = tmp_path / "output"

        a = Registrar.from_manifest(str(input_dir))
        a.update(
            yaml_strings={"turn2": "thing:\n- same_as: thing\n- description: second\n"},
            time=2,
        )
        a.export_manifest(output_dir=str(output_dir), collapse_time_dimension=False)

        with open(output_dir / "thing.yaml") as f:
            text = f.read()
        assert "first" in text
        assert "second" in text
