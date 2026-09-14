"""A YAML loader with YAML 1.2 float resolution.

PyYAML implements YAML 1.1, whose float pattern requires a decimal point before the
exponent. Under it, ``1e-6`` is a **string**, while ``1.0e-6`` is a float::

    >>> import yaml
    >>> yaml.safe_load("tol: 1e-6")["tol"]
    '1e-6'

That is a silent trap in a scientific input file, where ``1e-6``, ``1E-8`` and ``-1e-8``
are the natural way to write a tolerance. The user would see a puzzling type error at
best, and the surprise scales with how obviously correct their input looks.

:class:`AmaterasuLoader` re-registers the float resolver with the YAML 1.2 core-schema
pattern (the same one JSON uses), so all of those forms parse as numbers. Nothing else
about ``SafeLoader`` changes -- no arbitrary object construction is enabled.
"""

from __future__ import annotations

import re

import yaml

__all__ = ["AmaterasuLoader", "safe_load"]


class AmaterasuLoader(yaml.SafeLoader):
    """SafeLoader that resolves floats and booleans the way YAML 1.2 does."""


# YAML 1.2 core schema float, plus the .inf/.nan special values YAML 1.1 already had.
_FLOAT_PATTERN = re.compile(
    r"""^(?:
        [-+]?(?:[0-9][0-9_]*)\.[0-9_]*(?:[eE][-+]?[0-9]+)?
       |[-+]?\.[0-9][0-9_]*(?:[eE][-+]?[0-9]+)?
       |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
       |[-+]?\.(?:inf|Inf|INF)
       |\.(?:nan|NaN|NAN)
    )$""",
    re.VERBOSE,
)

# Drop the inherited YAML 1.1 float resolver before adding ours, or the 1.1 pattern still
# wins for the first characters it claims.
AmaterasuLoader.yaml_implicit_resolvers = {
    key: [(tag, regexp) for tag, regexp in resolvers if tag != "tag:yaml.org,2002:float"]
    for key, resolvers in yaml.SafeLoader.yaml_implicit_resolvers.items()
}
AmaterasuLoader.add_implicit_resolver(
    "tag:yaml.org,2002:float", _FLOAT_PATTERN, list("-+0123456789.")
)

# YAML 1.1 also resolves yes/no/on/off as booleans -- the "Norway problem". In a chemistry
# code that is not a curiosity: the element symbol No (nobelium) becomes False, so
# `hubbard: {species: {No: {U: 4.0}}}` silently loses the species. Restricting booleans to
# the YAML 1.2 core set (true/false) removes the collision. `symmetry: yes` then arrives as
# the string "yes" and is rejected by type validation with a message naming the fix, which
# is a far better outcome than mangling an element symbol.
_BOOL_PATTERN = re.compile(r"^(?:true|True|TRUE|false|False|FALSE)$")

AmaterasuLoader.yaml_implicit_resolvers = {
    key: [(tag, regexp) for tag, regexp in resolvers if tag != "tag:yaml.org,2002:bool"]
    for key, resolvers in AmaterasuLoader.yaml_implicit_resolvers.items()
}
AmaterasuLoader.add_implicit_resolver("tag:yaml.org,2002:bool", _BOOL_PATTERN, list("tTfF"))


def safe_load(stream):
    """``yaml.safe_load`` with YAML 1.2 float resolution.

    >>> safe_load("tol: 1e-6")["tol"]
    1e-06
    >>> safe_load("a: 1.0e-6\\nb: -1E8\\nc: 0.5\\nd: 5\\ne: true\\n")
    {'a': 1e-06, 'b': -100000000.0, 'c': 0.5, 'd': 5, 'e': True}

    The element symbol No stays a string rather than becoming False:

    >>> list(safe_load("No: 1"))
    ['No']
    """
    return yaml.load(stream, Loader=AmaterasuLoader)
