from setuptools import setup, find_packages
from os import path

if __name__ == "__main__":
    setup(
        name='IDPForge', 
        version='1.0.0',
        description='A generative model for sampling IDPs and proteins with IDRs.',
        long_description=None,
        author='Oufan Zhang',
        author_email='oz57@berkekey.edu',
        packages=find_packages(include=['esm*', 'idpforge*']),
        install_requires=[
            'torch',
            'pyyaml'
        ],
        license='MIT',
        keywords=[
            'Machine Learning', 'Protein Ensemble',
            'Structure Determination', "Generative Model",
        ],
    )