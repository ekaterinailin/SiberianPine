from setuptools import setup, find_packages

setup(
    name='siberianpine',
    version='0.0.1',
    url='https://github.com/ekaterinailin/siberianpine.git',
    author='Ekaterina Ilin',
    author_email='eilin@aip.de',
    description='Analyse statistical samples of flares.',
    packages=find_packages(),
    install_requires=['numpy', 'pandas', 'emcee', 'corner'],
)
