from setuptools import setup

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="simpledt",
    version="1.0.0",
    author="Anton Pechenko and ChatGPT",
    author_email="forpost78 aat gmail ddot com",
    description="A package for training and evaluating the DecisionTransformer model",
    long_description=long_description,
    long_description_content_type="text/markdown",
    # url="https://github.com/janedoe/decisiontransformer",
    packages=["simpledt"],
    package_dir={"": "src"},
    scripts=[
        "src/bin/train_dt",
        "src/bin/train_cem",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
    install_requires=[
        "torch>=1.6",
        "numpy>=1.19",
        "gymnasium",
        "tensorboard",
    ],
)
