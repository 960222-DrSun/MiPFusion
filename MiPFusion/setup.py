"""Editable-install shim so that `python setup.py develop` / `pip install -e .`
exposes the project's `models` package (matching the BasicSR convention).

Only the local `models` package is installed; the official `basicsr` package is
a normal external dependency and is never modified.
"""

from setuptools import find_packages, setup

setup(
    name='mipfusion',
    version='1.0.0',
    description='MiPFusion -- ICASSP paper reference implementation (BasicSR template)',
    packages=find_packages(exclude=('experiments', 'results', 'datasets', 'options')),
    include_package_data=True,
    install_requires=[],
    python_requires='>=3.8',
)
