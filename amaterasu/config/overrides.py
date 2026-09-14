"""Dotted-path overrides for the YAML input (``--set kpoints.grid='[4,4,4]'``).

Overrides are applied to the raw mapping *before* the dataclasses are built, so an
override with a bad type or an unknown key is reported by exactly the same validation
that checks the file itself -- there is no second, weaker code path.

The value is parsed as YAML, so ``--set`` handles every literal the file does::

    --set basis.ecut=80                  ->  int    80
    --set basis.ecut='80 Ry'             ->  str    '80 Ry'
    --set kpoints.grid='[4, 4, 4]'       ->  list   [4, 4, 4]
    --set system.spin=polarized          ->  str    'polarized'
    --set relax.cell=true                ->  bool   True
    --set occupations.nbands=null        ->  None
    --set system.hubbard.species.Ni.U=6  ->  nested mapping
"""

from __future__ import annotations

from typing import Any

import yaml

from ._yaml import safe_load

__all__ = ["apply_overrides", "parse_override"]


def parse_override(assignment: str) -> tuple[list[str], Any]:
    """Split ``"a.b.c=value"`` into ``(["a", "b", "c"], parsed_value)``."""
    key, separator, raw = assignment.partition("=")
    if not separator:
        raise ValueError(f"--set expects 'path.to.key=value', got {assignment!r}")
    path = [part for part in key.strip().split(".") if part]
    if not path:
        raise ValueError(f"--set is missing a key path in {assignment!r}")
    try:
        value = safe_load(raw)
    except yaml.YAMLError as exc:
        raise ValueError(f"cannot parse value in {assignment!r}: {exc}") from exc
    return path, value


def apply_overrides(data: dict, assignments: list[str]) -> dict:
    """Return a copy of ``data`` with every ``path=value`` assignment applied."""
    result = _deepcopy_mapping(data)
    for assignment in assignments:
        path, value = parse_override(assignment)
        node = result
        for depth, part in enumerate(path[:-1]):
            child = node.get(part)
            if not isinstance(child, dict):
                if child is not None:
                    traversed = ".".join(path[: depth + 1])
                    raise ValueError(
                        f"--set {assignment!r}: {traversed!r} is a value, not a section, "
                        "so it has no sub-keys"
                    )
                child = {}
                node[part] = child
            node = child
        node[path[-1]] = value
    return result


def _deepcopy_mapping(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _deepcopy_mapping(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_deepcopy_mapping(v) for v in value]
    return value
