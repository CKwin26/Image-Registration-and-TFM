from setuptools import setup, find_packages

setup(
    name='ir',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'numpy>=1.16.2',
        'matplotlib>=3.0',
        'pandas>=1.1.5',
        'scipy>=1.7.0',
        'scikit-image>=0.14.2',
        'openpiv>=0.23.4',
        'tqdm>=4.26.0',
        'natsort>=5.1.0',
        'dill>=0.2.9',
        'roipoly>=0.5.2',
        'openpyxl',
        'pyyaml',
        'matplotlib-scalebar'
    ],
)
