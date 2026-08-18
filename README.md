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
