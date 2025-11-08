"""
Flightmare Modern RL Setup
Modern reinforcement learning using PyTorch + Stable-Baselines3 + Gymnasium
"""
import os
from setuptools import setup, find_packages

# Read version from __init__.py
version = {}
with open(os.path.join(os.path.dirname(__file__), "__init__.py")) as f:
    for line in f:
        if line.startswith("__version__"):
            exec(line, version)
            break

# Read requirements from requirements.txt
with open(os.path.join(os.path.dirname(__file__), "requirements.txt")) as f:
    requirements = [
        line.strip() 
        for line in f 
        if line.strip() and not line.startswith("#")
    ]

setup(
    name='flightrl_modern',
    version=version.get("__version__", "0.0.1"),
    author='Ahmed Ali',
    author_email='ali.a@aucegypt.edu',
    description='Modern reinforcement learning for Flightmare using PyTorch + Stable-Baselines3',
    long_description=open(os.path.join(os.path.dirname(__file__), "README.md")).read(),
    long_description_content_type='text/markdown',
    url='https://github.com/uzh-rpg/flightmare',
    packages=find_packages(exclude=['tests', 'tests.*', 'examples', 'examples.*']),
    install_requires=requirements,
    python_requires='>=3.8',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
    ],
    extras_require={
        'dev': [
            'pytest>=7.0.0',
            'pytest-cov>=4.0.0',
            'black>=23.0.0',
            'flake8>=6.0.0',
        ],
    },
)
