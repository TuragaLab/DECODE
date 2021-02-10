import copy
from typing import Iterable

import yaml


def convert_mixed_list(mix) -> dict:
    """Convert a list of elements and dicts to list of dicts (with None values for non dicts)"""

    mix_dict = dict.fromkeys([k for k in mix if not isinstance(k, dict)])
    for k in [k for k in mix if isinstance(k, dict)]:  # limit to dicts
        mix_dict.update(k)

    return mix_dict


def convert_to_spec(package):
    """Convert yaml style '=' to '==' as in spec."""
    if '=' not in package or '<' in package or '>' in package:
        return package
    return package.replace('=', '==')


def add_update_package(deps: dict, update: dict, level: Iterable) -> dict:
    """Adds or updates a package"""

    update = {k: v for k,v in update.items() if k in level}  # limit to active levels
    for n_deps in update.values():  # loop over level
        for k, v in n_deps.items():
            if k in deps:
                deps[v] = deps.pop(k)
            else:
                deps[k] = None

    return deps


def conda(run_deps, dev_deps, doc_deps, channels, level, mode):
    """
    Generate conda environment or spec file.

    Args:
        run_deps:
        dev_deps:
        doc_deps:
        channels:
        level: which level, i.e. 'run' or 'dev' or ('dev', 'docs')
        mode: either environment file (env) or specs (txt)

    """
    level = (level, ) if not isinstance(level, tuple) else level

    name = 'decode'

    deps = dict.fromkeys(run_deps)
    if 'dev' in level:
        deps.update(dict.fromkeys(dev_deps))
        name += '_dev'
    if 'docs' in level:
        deps.update(dict.fromkeys(doc_deps))
        name += '_docs'

    if mode == 'txt':
        return [convert_to_spec(o) for o in deps]

    elif mode == 'env':
        out = {
            'name': name,
            'channels': channels,
            'dependencies': list(deps.keys())
        }

        return out


def conda_meta(run_deps, meta):

    deps = conda(run_deps, None, None, None, 'run', 'env')['dependencies']
    deps = dict.fromkeys(deps)

    # apply meta changes to environment
    deps = add_update_package(deps, meta, ('run', ))

    build = copy.copy(meta)
    build['run'] = list(deps.keys())

    return build


def pip(run_deps, dev_deps, doc_deps, pip, level):

    level = (level,) if not isinstance(level, tuple) else level

    deps = dict()
    if 'run' in level:
        deps.update(dict.fromkeys(run_deps))
    if 'dev' in level:
        deps.update(dict.fromkeys(dev_deps))
    if 'docs' in level:
        deps.update(dict.fromkeys(doc_deps))

    deps = add_update_package(deps, pip, level)

    return [convert_to_spec(p) for p in deps]


def parse_dependency(path) -> dict:

    with open(path, 'r') as stream:
        data = yaml.safe_load(stream)

    for k, v in data['pip'].items():
        data['pip'][k] = convert_mixed_list(v)

    data['conda-build']['run'] = convert_mixed_list(data['conda-build']['run'])

    return data

