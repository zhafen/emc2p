"""Hand-checked expected value for the etl.load_manifest.raw_entity_first_data
node, for the examples/networks example (net_AB.yaml half of it)."""

raw_entity_first_data = {
    "examples/networks/net_AB.yaml": {
        "A": ["node"],
        "B": ["node"],
        "AB": [
            {"link": {"source": "A", "target": "B", "link_type": "alpha"}},
        ],
    },
}

# A negative case: AB is a link from A to B, not B to A, so this should NOT
# be found as a subset of the actual output.
incorrect_raw_entity_first_data = {
    "examples/networks/net_AB.yaml": {
        "AB": [
            {"link": {"source": "B", "target": "A", "link_type": "alpha"}},
        ],
    },
}
