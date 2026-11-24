"""The path filters of the CI workflows (.github/workflows/tests-<chunk>.yml). Each workflow runs one chunk of the tests (see
run_tests.chunk_of) and only when a file that the chunk covers changes in a push. A filter that misses a file would let a change
go untested; this checks, for every chunk, that its filter lists every source file that the tests of the chunk import (directly,
through other modules, lazily inside a function, or by running a module in a subprocess), the tests themselves, and the files that
configure the run. Run with:  pytest tests/utils/test_ci_paths.py

When it fails after a change, add the files it names to the `paths:` of that workflow (or move the test to another chunk in
run_tests.chunk_of).
"""
import ast
import importlib.util
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('run_tests', ROOT / 'run_tests.py')
run_tests = importlib.util.module_from_spec(spec)
spec.loader.exec_module(run_tests)

CONFIG = ['run_tests.py', 'requirements.txt', 'pyproject.toml', 'conftest.py']        # a change of these affects every test


def module_file(name):
    base = ROOT.joinpath(*name.split('.'))
    if base.with_suffix('.py').is_file():
        return str(base.with_suffix('.py').relative_to(ROOT))
    if (base / '__init__.py').is_file():
        return str((base / '__init__.py').relative_to(ROOT))
    return None


def imports_of(path):
    """the pylattice files that a file imports (import statements anywhere in it; in a test also the module run with -m)"""
    source = (ROOT / path).read_text()
    names = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            names |= {a.name for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
            names |= {node.module + '.' + a.name for a in node.names}
    if path.startswith('tests/'):
        names |= set(re.findall(r"-m', '(pylattice(?:\.\w+)+)'", source))
    found = set()
    for name in names:
        parts = name.split('.')
        if parts[0] != 'pylattice':
            continue
        for k in range(len(parts), 0, -1):                                           # the longest prefix that is a module
            f = module_file('.'.join(parts[:k]))
            if f:
                found.add(f)
                break
    return found


def closure(tests):
    """every pylattice file that the tests need, with the __init__.py of its packages"""
    seen, todo = set(), []
    for t in tests:
        todo += imports_of(t)
    while todo:
        f = todo.pop()
        if f in seen:
            continue
        seen.add(f)
        todo += imports_of(f)
        d = Path(f).parent
        while str(d).startswith('pylattice'):
            if (ROOT / d / '__init__.py').is_file():
                todo.append(str(d / '__init__.py'))
            d = d.parent
    return seen


def workflow(chunk):
    return ROOT / '.github' / 'workflows' / 'tests-{}.yml'.format(chunk)


def paths_of(chunk):
    """the `paths:` of the push trigger of a workflow (read as text: no YAML library is needed)"""
    out, inside = [], False
    for line in workflow(chunk).read_text().splitlines():
        if re.match(r'^\s+paths:\s*$', line):
            inside = True
        elif inside:
            m = re.match(r"^\s+- '([^']+)'\s*(#.*)?$", line)
            if m:
                out.append(m.group(1))
            elif line.strip() and not line.strip().startswith('#'):
                break
    return out


@pytest.mark.parametrize('chunk', run_tests.CHUNKS)
def test_the_filter_of_each_workflow_covers_what_its_tests_import(chunk):
    tests = run_tests.files_of_chunk(chunk)
    needed = closure(tests) | set(tests) | set(CONFIG) | {str(workflow(chunk).relative_to(ROOT))}
    missing = sorted(needed - set(paths_of(chunk)))
    assert not missing, 'the push filter of {} misses: {}'.format(workflow(chunk).name, missing)


@pytest.mark.parametrize('chunk', run_tests.CHUNKS)
def test_the_filter_lists_only_files_that_exist_and_each_workflow_runs_its_chunk(chunk):
    paths = paths_of(chunk)
    assert paths and len(paths) == len(set(paths))
    assert [p for p in paths if not (ROOT / p).is_file()] == []
    text = workflow(chunk).read_text()
    assert 'python run_tests.py --chunk {} '.format(chunk) in text or 'python run_tests.py --chunk {}\n'.format(chunk) in text
    assert 'workflow_dispatch' in text and 'timeout-minutes' in text and 'concurrency' in text


def test_the_chunks_are_filtered_differently_and_shared_code_runs_all_of_them():
    # a change that only touches one theory runs only the chunks that use it
    core, gauge, spin = (set(paths_of(c)) for c in run_tests.CHUNKS)
    assert 'pylattice/theories/gauge/sun.py' in gauge and 'pylattice/theories/gauge/sun.py' not in core | spin
    assert 'pylattice/theories/spin/ising_meas.py' in spin and 'pylattice/theories/spin/ising_meas.py' not in gauge
    assert 'pylattice/core/groups/su.py' in gauge and 'pylattice/core/groups/su.py' not in core | spin
    assert 'pylattice/core/grid.py' in core & gauge & spin                                   # shared code runs everything
    assert 'README.md' not in core | gauge | spin and 'docs/mylogo.png' not in core | gauge | spin   # documentation runs nothing
