#!/usr/bin/env python

from setuptools import setup, find_packages

setup(
    name='project',
    version='0.0.0',
    description='Jeopardy-like Contrastive Learning',
    author='',
    author_email='',
    url='https://github.com/shashank2000/JeopardyContrastive',
    install_requires=['pytorch-lightning'],
    packages=find_packages(),
)

