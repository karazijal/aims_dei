import setuptools


setuptools.setup(
    name="AIMS_DEI_LAB_CODE", # Replace with your own username
    version="0.0.1",
    author="Laurynas Karazija",
    author_email="laurynas@robots.ox.ac.uk",
    description="Code for AIMS DEI Labs",
    long_description="Code for AIMS DEI Labs",
    long_description_content_type="text/markdown",
    url="http://www.robots.ox.ac.uk/~mosb/teaching/AIMS_CDT/CDT_estimation_inference_lab.pdf",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=[
       'requests',
       'matplotlib',
       'numpy',
       'scipy',
       'pandas',
       'jax'
    ]
)