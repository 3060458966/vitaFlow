"""Install vitaFlow."""

from setuptools import find_packages
from setuptools import setup

with open('requirements.txt') as f:
    install_requires = f.read().splitlines()

setup(
    name='vitaFlow',
    version='0.1.0',
    description='VideoImageTextAudioFlow',
    author='Imaginea',
    author_email='mageswaran.dhandapani@imaginea.com',
    url='http://github.com/imaginea/vitaFlow',
    license='Apache 2.0',
    packages=find_packages(),
    package_data={
    },
    scripts=[
        'vitaflow/bin/vf_engine.py'
    ],
    install_requires=install_requires,
    extras_require={
        'tensorflow': ['tensorflow>=1.12.0'],
        'tensorflow_gpu': ['tensorflow-gpu>=1.12.0'],
        'tensorflow-hub': ['tensorflow-hub>=0.1.1'],
        'tests': [
            'absl-py',
            'pytest>=3.8.0',
            'mock',
            'pylint',
            'jupyter',
            'gsutil',
            'matplotlib']
            # Need atari extras for Travis tests, but because gym is already in
            # install_requires, pip skips the atari extras, so we instead do an
            # explicit pip install gym[atari] for the tests.
            # 'gym[atari]',

    },
    classifiers=[
        'Development Status :: 1 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Enginerring/Research',
        'License :: OSI Approved :: Apache Software License',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
    dependency_links=[
    ],
    keywords='tensorflow machine learning',
)
