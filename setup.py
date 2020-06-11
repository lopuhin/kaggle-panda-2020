from setuptools import setup, find_packages


setup(
    name='panda',
    packages=find_packages(),
    entry_points={
        'console_scripts': [
            'panda-train=panda.train:main',
        ],
    },
)
