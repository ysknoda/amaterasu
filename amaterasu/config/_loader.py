"""Type-driven construction of the configuration dataclasses from parsed YAML.

The schema in :mod:`amaterasu.config.schema` is expressed purely as dataclasses with
standard type hints. This module turns a nested ``dict`` (straight out of ``yaml.safe_load``)
into those dataclasses, validating as it goes:

* unknown keys are reported (with a "did you mean ...?" suggestion) rather than ignored --
  a silently ignored ``ecutwfc:`` typo means the run quietly used the default cutoff;
* ``Literal[...]`` hints become the list of allowed values;
* every problem is accumulated and raised once, so one ``--dry-run`` shows the whole
  list of mistakes instead of one per edit-rerun cycle.
"""

from __future__ import annotations

import dataclasses
import difflib
import functools
import operator
import types
import typing
from typing import Any, Literal, Union, get_args, get_origin

__all__ = ["ConfigError", "ErrorLog", "build_dataclass", "to_plain_dict"]


class ConfigError(ValueError):
    """Raised when a configuration cannot be built. Carries every problem found.

    A single string is accepted as well as a list. Without that, ``ConfigError("...")``
    would iterate the message character by character and report one "problem" per letter
    -- a small trap that is worth closing at the source rather than at every call site.
    """

    def __init__(self, problems: list[str] | str, source: str = "input"):
        self.problems = [problems] if isinstance(problems, str) else list(problems)
        if len(self.problems) == 1:
            super().__init__(self.problems[0])
            return
        header = f"{len(self.problems)} problem(s) in {source}:"
        super().__init__("\n".join([header, *(f"  {p}" for p in self.problems)]))


class ErrorLog:
    """Accumulates configuration problems so they can all be reported at once."""

    def __init__(self) -> None:
        self.problems: list[str] = []

    def add(self, path: str, message: str) -> None:
        self.problems.append(f"{path}: {message}" if path else message)

    def raise_if_any(self, source: str = "input") -> None:
        if self.problems:
            raise ConfigError(self.problems, source)


def _join(path: str, key: str) -> str:
    return f"{path}.{key}" if path else key


def _type_name(hint: Any) -> str:
    if get_origin(hint) is Literal:
        return " | ".join(repr(a) for a in get_args(hint))
    return getattr(hint, "__name__", str(hint).replace("typing.", ""))


def _suggest(key: str, valid: typing.Iterable[str]) -> str:
    close = difflib.get_close_matches(key, list(valid), n=1, cutoff=0.6)
    return f" (did you mean {close[0]!r}?)" if close else ""


def _is_optional(hint: Any) -> tuple[bool, Any]:
    """Return ``(optional, hint_without_None)`` for ``X | None`` style hints."""
    if get_origin(hint) in (Union, types.UnionType):
        args = [a for a in get_args(hint) if a is not type(None)]
        if len(args) != len(get_args(hint)):
            # Rebuild the union at runtime; Union[tuple] would be an annotation form.
            return True, functools.reduce(operator.or_, args)
    return False, hint


def _coerce(value: Any, hint: Any, path: str, log: ErrorLog) -> Any:
    """Convert ``value`` to ``hint``, logging a problem and returning None on failure."""
    optional, hint = _is_optional(hint)
    if value is None:
        if not optional:
            log.add(path, "may not be null")
        return None

    origin = get_origin(hint)

    if origin is Literal:
        allowed = get_args(hint)
        if value in allowed:
            return value
        # Be forgiving about case for string choices; the YAML is human-written.
        if isinstance(value, str):
            lowered = {str(a).lower(): a for a in allowed}
            if value.lower() in lowered:
                return lowered[value.lower()]
        log.add(path, f"{value!r} is not one of: {', '.join(repr(a) for a in allowed)}")
        return None

    if origin in (Union, types.UnionType):
        # Try each member in order; used for "number or string with a unit" fields.
        for member in get_args(hint):
            probe = ErrorLog()
            result = _coerce(value, member, path, probe)
            if not probe.problems:
                return result
        log.add(path, f"expected {_type_name(hint)}, got {type(value).__name__} {value!r}")
        return None

    if dataclasses.is_dataclass(hint):
        if not isinstance(value, dict):
            log.add(path, f"expected a mapping of settings, got {type(value).__name__}")
            return None
        return build_dataclass(hint, value, path, log)

    if origin in (list, tuple):
        if not isinstance(value, (list, tuple)):
            log.add(path, f"expected a list, got {type(value).__name__} {value!r}")
            return None
        (item_hint,) = get_args(hint) or (Any,)
        return [_coerce(v, item_hint, f"{path}[{i}]", log) for i, v in enumerate(value)]

    if origin is dict:
        if not isinstance(value, dict):
            log.add(path, f"expected a mapping, got {type(value).__name__} {value!r}")
            return None
        key_hint, value_hint = get_args(hint) or (Any, Any)
        return {
            _coerce(k, key_hint, f"{path}<key>", log): _coerce(
                v, value_hint, _join(path, str(k)), log
            )
            for k, v in value.items()
        }

    if hint is Any:
        return value

    if hint is bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str) and value.lower() in ("yes", "no", "on", "off"):
            # amaterasu resolves booleans per YAML 1.2 so that the element symbol No is
            # not read as False; point the user at the spelling that works.
            log.add(path, f"write true/false rather than {value!r} (YAML 1.2 booleans)")
            return None
        log.add(path, f"expected true/false, got {type(value).__name__} {value!r}")
        return None

    if hint is float:
        # bool is an int subclass; never accept it as a number.
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)
        log.add(path, f"expected a number, got {type(value).__name__} {value!r}")
        return None

    if hint is int:
        if isinstance(value, int) and not isinstance(value, bool):
            return value
        log.add(path, f"expected an integer, got {type(value).__name__} {value!r}")
        return None

    if hint is str:
        if isinstance(value, str):
            return value
        log.add(path, f"expected a string, got {type(value).__name__} {value!r}")
        return None

    if isinstance(value, hint):
        return value
    log.add(path, f"expected {_type_name(hint)}, got {type(value).__name__} {value!r}")
    return None


def build_dataclass(cls: type, data: dict, path: str, log: ErrorLog):
    """Build an instance of the dataclass ``cls`` from ``data``, logging every problem."""
    hints = typing.get_type_hints(cls)
    fields = {f.name: f for f in dataclasses.fields(cls)}

    for key in data:
        if key not in fields:
            log.add(_join(path, str(key)), f"unknown setting{_suggest(str(key), fields)}")

    kwargs = {}
    for name, field in fields.items():
        if name in data:
            kwargs[name] = _coerce(data[name], hints[name], _join(path, name), log)
        elif field.default is not dataclasses.MISSING:
            kwargs[name] = field.default
        elif field.default_factory is not dataclasses.MISSING:  # type: ignore[misc]
            kwargs[name] = field.default_factory()  # type: ignore[misc]
        else:
            log.add(_join(path, name), "is required but was not given")
            kwargs[name] = None
    return cls(**kwargs)


def to_plain_dict(obj: Any) -> Any:
    """Recursively convert dataclasses to plain dicts for YAML serialisation."""
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {f.name: to_plain_dict(getattr(obj, f.name)) for f in dataclasses.fields(obj)}
    if isinstance(obj, dict):
        return {k: to_plain_dict(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_plain_dict(v) for v in obj]
    return obj
