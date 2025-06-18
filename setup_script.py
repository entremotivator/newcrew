from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="ai-property-analyst",
    version="2.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Advanced AI-powered real estate analysis with multiple specialized crews",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/ai-property-analyst",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Real Estate Professionals",
        "Topic :: Office/Business :: Financial :: Investment",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.9",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0",
            "black>=23.0",
            "flake8>=6.0",
            "mypy>=1.0",
            "pre-commit>=3.0",
        ],
        "deployment": [
            "gunicorn>=20.0",
            "nginx",
            "supervisor",
        ],
    },
    entry_points={
        "console_scripts": [
            "property-analyst=property_analyst:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.toml", "*.yaml", "*.yml", "*.json"],
    },
)