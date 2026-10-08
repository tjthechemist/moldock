from setuptools import setup, find_packages

setup(
    name="moldock",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.20.0",
    ],
    extras_require={
        "full": ["scipy", "vina", "rdkit", "meeko", "biopython"],
        "dev": ["pytest"],
    },
    entry_points={
        "console_scripts": [
            "moldock=moldock.cli.main:main",
        ],
    },
)
