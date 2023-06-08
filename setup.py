from setuptools import setup, find_packages

setup(
    name="auto_test",
    version="0.1.0",
    package_dir={"": "app"},
    packages=find_packages(where="app"),
    install_requires=[
        "openai",
        "pygments",
        "colorama"
    ],
)