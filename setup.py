from setuptools import setup


def long_description():
	with open('README.md') as f:
		return f.read()


setup(
    name="vectorkit",
    version="0.1.3",
	packages=["vectorkit"],
    description="Vector Kit seeks to make vector arithmetic simple for everyone. "
	"It may serve as a utility in a large ecosystem of scientific libraries or, "
	"more simply, as a toy to be played with to understand Vector math.",
	long_description=long_description(),
	long_description_content_type="text/markdown",
    keywords=["AI", "vectors", "algebra", "geometry"],
    author="Victor Mawusi Ayi",
    author_email="ayivima@hotmail.com",
    url="https://github.com/ayivima/vectorkit",
    license="MIT",
    python_requires=">=3.3",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python",
		"Programming Language :: Python :: 3.4",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3 :: Only",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
	entry_points = {
        'console_scripts': ['vectorkit=vectorkit.vectortools:main'],
    }
)
