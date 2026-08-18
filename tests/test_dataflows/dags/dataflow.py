"""Hamilton dataflow module: functions here define the DAG nodes."""

import ibis

from emc2p.registry import Registry


def description_table(registry: Registry) -> ibis.Table:
    return registry.get("description")


def entity_summary(description_table: ibis.Table) -> ibis.Table:
    """Get entity summaries from description table."""
    return description_table.select("entity_id", description="value")
