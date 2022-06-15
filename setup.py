from setuptools import setup, find_packages

with open("README.md", "r") as f:
    my_long_description = f.read()

with open('LICENSE') as f:
    my_license = f.read()

with open('requirements.txt') as f:
    my_requirements = f.read().splitlines()


def main():
    setup(
        name="otto",
        version="1.1",
        author="A. Loisy, C. Eloy",
        author_email="aurore.loisy@gmail.com, eloy@irphe.univ-mrs.fr",
        description="OTTO (Odor-based Target Tracking Optimization): "
                    "a Python package to simulate, solve and visualize the source-tracking POMDP",
        long_description=my_long_description,
        long_description_content_type="text/markdown",
        url='https://github.com/C0PEP0D/otto',
        license=my_license,
        packages=find_packages(),
        classifiers=[
            "Programming Language :: Python :: 3",
            "License :: OSI Approved :: MIT License",
            "Operating System :: OS Independent",
        ],
        python_requires=">=3.8",
        install_requires=my_requirements,
    )


if __name__ == '__main__':
    main()
