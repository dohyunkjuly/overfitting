from setuptools import setup, find_packages

setup(
    name='Overfitting',
    version='0.0.1',
    packages=find_packages(),
    license='MIT',
    description='A Robust Futures CryptoCurrency Backtesting Library.',
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
        'Intended Audience :: Crypto Traders',
        'License :: OSI Approved :: MIT License',
        'Framework :: Jupyter',
        'Programming Language :: Python',
        'Operating System :: OS Independent',
        'Topic :: Investment'
    ],
    keywords = [
        'algo', 'bitcoin', 'ethereum', 'crypto', 'cryptocurrency',
        'crypto derivatives', 'futures', 'finance', 'quantitative',
        'liquidation', 'solana', 'systematic', 'quant', 'trading'
    ]
)
