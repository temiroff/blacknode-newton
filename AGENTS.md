# blacknode-newton contributor instructions

This package is an independently versioned Blacknode extension.

- Keep the Newton runtime independent of a concrete viewer or ROS transport. Viewer providers register through `nodes/viewer_contract.py`; transports issue commands through the managed session API.
- A provider must report a structured unavailable state when its optional dependency is absent.
- Keep every session disarmed by default. Clamp targets to USD joint limits and rate-limit motion before writing Newton controls.
- Keep self-collision opt-in. Asset-specific collision limitations belong in tests or asset documentation, not the generic scene contract.
- Do not add manifest roadmap placeholders. A declared component must contain working code and tests.
- Preserve model attribution in `NOTICE` when replacing or modifying the bundled USD.
- Run `python -m unittest discover -s tests -v` from this package after changes. Run Blacknode package/workflow validation when node contracts or templates change.
