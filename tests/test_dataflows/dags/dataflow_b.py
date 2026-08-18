"""Second Hamilton dataflow module for testing multiple modules."""

import ibis


def entity_count(description_table: ibis.Table) -> ibis.Table:
    """Count entities in the description table."""
    return description_table.group_by([]).aggregate(n=description_table.count())
