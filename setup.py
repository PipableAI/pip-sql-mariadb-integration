from setuptools import find_packages, setup

__version__ = "0.0.1"

with open("README.md", "r", encoding="UTF-8") as f:
    long_description = f.read()

setup(
    name="pip_sql_performance_schema",
    version=__version__,
    packages=find_packages(),
    long_description=long_description,
    long_description_content_type="text/markdown",
    install_requires=[
        "transformers",
        "torch",
        "pymysql",
        "huggingface_hub"
    ],
    package_data={
        "pip_sql_performance_schema": ["data/*"],
    },
)
