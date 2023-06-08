from setuptools import setup, find_packages

setup(
    name="auto_test",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "openai",
        "pygments",
        "colorama"
    ],
)