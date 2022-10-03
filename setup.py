from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()
with open("requirements.txt", "r", encoding="utf-8") as f:
    requirements = f.read()

setup(
    name="transformer-pytorch",
    version="0.1.0",
    author="Bhimraj Yadav",
    author_email="bhimrajyadav977@gmail.com",
    description="Transformer Implementation from scratch using PyTorch.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/bhimrazy/transformer-pytorch",
    packages=find_packages(),
    py_modules=['main'],
    install_requires=[requirements],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    entry_points='''
        [console_scripts]
        py-transformer=main:cli
    ''',
    python_requires='>=3.6',
)
