from setuptools import setup, find_packages

# Read the contents of README file
with open("README.md", encoding="utf-8") as f:
    long_description = f.read()

# Read requirements from requirements.txt
with open("requirements.txt", encoding="utf-8") as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]

setup(
    name="whatsapp-reality",
    version="0.1.0",
    author="Abdul",
    author_email="rasoolas2003@gmail.com",
    description="A comprehensive WhatsApp chat analysis library",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Abdul1028/whatsapp-reality",
    project_urls={
        "Bug Tracker": "https://github.com/Abdul1028/whatsapp-reality/issues",
        "Documentation": "https://github.com/Abdul1028/whatsapp-reality#readme",
        "Source Code": "https://github.com/Abdul1028/whatsapp-reality",
    },
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Communications :: Chat",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Topic :: Text Processing :: General",
    ],
    python_requires=">=3.7",
    install_requires=requirements,
    include_package_data=True,
    package_data={
        "whatsapp_reality": ["data/*.txt"],
    },
    keywords="whatsapp chat analysis nlp sentiment emoji timeline visualization",
    tests_require=[
        'pytest>=6.0.0',
        'pytest-cov>=2.0.0',
    ],
    test_suite='tests',
) 