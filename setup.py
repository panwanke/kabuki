from distutils.core import setup

setup(
    name="kabuki",
    version="0.6.5RC4",
    author="Thomas V. Wiecki, Imri Sofer, revised by Wanke Pan",
    author_email="thomas.wiecki@gmail.com",
    url="https://gitee.com/epool/kabuki",
    packages=["kabuki"],
    description="kabuki is a python toolbox that allows easy creation of hierarchical bayesian models for the cognitive sciences.",
    install_requires=['NumPy >= 1.6.0', 'pymc >= 2.3.6', 'pandas >= 0.12.0', 'matplotlib >= 1.0.0', 'SciPy >= 0.6.0', 'cloudpickle >= 2.0.0', 'arviz >= 0.11.0', 'joblib >= 1.2.0'],
    setup_requires=['NumPy >= 1.6.0', 'pymc >= 2.3.6', 'pandas >= 0.12.0', 'matplotlib >= 1.0.0', 'SciPy >= 0.6.0', 'cloudpickle >= 2.0.0', 'arviz >= 0.11.0', 'joblib >= 1.2.0']
)
