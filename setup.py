from setuptools import setup, find_packages
from pathlib import Path

this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name='overfitting',
    version='0.0.2',
    packages=find_packages(),
    license='MIT',
    description='A Robust Futures CryptoCurrency Backtesting Library.',
    long_description=long_description,
    long_description_content_type='text/markdown',  # 👈 This tells PyPI it's markdown

    author='Dohyun Kim',
    author_email='dohyun.k.july@gmail.com',
    author_github_link='https://github.com/dohyunkjuly',
    install_requires=[
        'numpy >= 1.17.0',
        'pandas >= 0.25.0',
        'seaborn',
        'matplotlib',
        'scipy',
    ],
    extra_requires={
        'examples': [
            'notebook', 
            'ipykernel', 
            'ipython'
        ],
        'dev': [
            'black', 
            'flake',
            'twine>=4.0.2'
        ],
    },
    classifiers=[
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Framework :: Jupyter',
        'Programming Language :: Python',
        'Operating System :: OS Independent',
        'Topic :: Office/Business :: Financial :: Investment',
    ],
    keywords = [
        'algo', 'bitcoin', 'ethereum', 'crypto', 'cryptocurrency',
        'crypto derivatives', 'futures', 'finance', 'quantitative',
        'liquidation', 'solana', 'systematic', 'quant', 'trading'
    ]
)
