from setuptools import setup, find_packages


setup(
    name='tusimple_duc',
    version='1.0.0',
    author='Pengfei Chen & Panqu Wang',
    description='semantic segmentation module on the Cityscapes dataset',
    install_requires=['configparser', 'numpy', 'Pillow'],
    url='https://github.com/TuSimple/TuSimple-DUC',
    packages=find_packages(),
)
