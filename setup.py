from distutils.core import setup

setup(
    name="kabuki",
    version="0.6.5RC4",
    author="Thomas V. Wiecki, Imri Sofer, revised by Wanke Pan",
    author_email="thomas.wiecki@gmail.com",
    url="https://github.com/panwanke/kabuki",
    packages=["kabuki"],
    description="kabuki is a python toolbox that allows easy creation of hierarchical bayesian models for the cognitive sciences.",
    install_requires=['NumPy >= 2, < 3', 'pymc >= 2.3.6, < 3', 'pandas >= 0.12.0', 'matplotlib >= 1.0.0', 'SciPy >= 0.6.0', 'cloudpickle >= 2.0.0', 'arviz >= 1.1.0', 'joblib >= 1.2.0'],
    setup_requires=['NumPy >= 2, < 3', 'pymc >= 2.3.6, < 3', 'pandas >= 0.12.0', 'matplotlib >= 1.0.0', 'SciPy >= 0.6.0', 'cloudpickle >= 2.0.0', 'arviz >= 1.1.0', 'joblib >= 1.2.0']
)
