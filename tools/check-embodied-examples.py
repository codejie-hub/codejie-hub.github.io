#!/usr/bin/env python3
"""Execute the exact Python examples published in the embodied learning series.

Requires NumPy. Only executes code from the five explicitly named local posts.
"""
from pathlib import Path
from contextlib import redirect_stdout
import io
import json
import re
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.stdout.reconfigure(encoding='utf-8')
SLUGS = ['imitation-learning', 'grasp-geometry', 'contact-control', 'world-models', 'offline-rl']
results = []
for slug in SLUGS:
    path = ROOT / '_posts' / f'2026-10-24-embodied-{slug}.md'
    text = path.read_text(encoding='utf-8')
    blocks = re.findall(r'^```python\n(.*?)^```', text, re.M | re.S)
    assert len(blocks) == 1, f'{path.name}: expected one complete executable example'
    output = io.StringIO()
    namespace = {'__name__': '__tutorial_example__'}
    with redirect_stdout(output):
        exec(compile(blocks[0], str(path), 'exec'), namespace)
    if slug == 'imitation-learning':
        assert np.isclose(namespace['before'][-1, 1], 0.2)
        assert np.isclose(namespace['after'][-1, 1], 0.2 * 0.5**20)
    # The assertions inside the other published examples validate their outcomes.
    results.append({'post': path.name, 'output': output.getvalue().strip(), 'passed': True})
    print(f'PASS {path.name}: {output.getvalue().strip()}')

# Independently verify the numerical quantities stated in the prose.
u, v, z, fx, fy = 380., 270., .6, 600., 600.
jacobian = np.array([[z/fx, 0, (u-320)/fx], [0, z/fy, (v-240)/fy], [0, 0, 1]])
std_mm = np.sqrt(np.diag(jacobian @ np.diag([1., 1., .005**2]) @ jacobian.T)) * 1000
assert np.allclose(std_mm, [1.118, 1.031, 5.], atol=.001)
n, p, critical = 20, .8, 1.96
center = (p + critical**2/(2*n))/(1+critical**2/n)
half = critical*np.sqrt(p*(1-p)/n+critical**2/(4*n*n))/(1+critical**2/n)
assert np.allclose([center-half,center+half],[.584,.919],atol=.001)
print('PASS: depth uncertainty and Wilson interval match the prose.')
target=ROOT/'.qa/embodied-examples.json';target.parent.mkdir(parents=True,exist_ok=True)
target.write_text(json.dumps(results,ensure_ascii=False,indent=2),encoding='utf-8')
