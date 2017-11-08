from setuptools import setup, find_packages


setup(
    name='tusimple_duc',
    version='1.0.0',
    author='Pengfei Chen',
    description='accurate segmentation module on Cityscapes data',
    install_requires=['configparser', 'numpy', 'Pillow'],
    url='https://github.com/GrassSunFlower/semantic-segmentation-duc-hdc',
    packages=find_packages(),
)
