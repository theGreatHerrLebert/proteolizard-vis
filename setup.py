from setuptools import setup

with open("README.md", "r") as fh:
    ld = fh.read()

setup(
    name='pyproteolizard-vis',
    version='0.0.1',
    description='a collection of python classes and widgets for display of all aquisitions generated by timsTOF',
    packages=['pyproteolizard-vis',],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v2 or later (GPLv3+)",
        "Operating System :: Linux"
    ],
    long_description=ld,
    long_description_content_type="text/markdown",
    install_requires=[
        "pyproteolizard >= 0.1",
        "pandas >=1.1",
        "hdbscan",
        "pandas",
        "sklearn",
        "plotly",
        "numpy",
        "ipython",
        "jupyter"
    ]
)