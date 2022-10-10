from setuptools import setup, find_packages

setup(
    name='si',
    version='0.0.1',
    python_requires='>=3.7',
    package_dir={'': 'src'},
    packages=find_packages('src'),
    zip_safe=False,
    install_requires=['numpy', 'pandas', 'scipy', 'matplotlib'],
    author='Rui Gomes',
    author_email='pg45970@alunos.uminho.pt',
    description='Sistemas inteligentes',
    license='Apache License Version 2.0',
    keywords='',
)
