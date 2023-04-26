import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name = "kronbinations",
    version         = "1.3",
    author = "Michael Schilling",
    author_email = "michael@ntropic.de",
    description  = "kronbinations is used to remove nested loops and perform parameter sweeps.",
    long_description = long_description,
    long_description_content_type = "text/markdown",
    url = "https://github.com/Ntropic/kronbinations/archive/refs/tags/v1.3.tar.gz",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    install_requires=["numpy", "tqdm"],
    python_requires=">=3.6",
)
