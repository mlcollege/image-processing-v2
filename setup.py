import sys
from setuptools.command.test import test as TestCommand
from setuptools import setup, find_packages


class PyTest(TestCommand):
    user_options = [('pytest-args=', 'a', "Arguments to pass to pytest")]

    def initialize_options(self):
        TestCommand.initialize_options(self)
        self.pytest_args = ''

    def run_tests(self):
        import shlex
        import pytest
        errno = pytest.main(shlex.split(self.pytest_args))
        sys.exit(errno)


setup(
    name='image_processing_workshop',
    version='1.0',
    description='Workshop custom tools',
    author='Adam Kolar',
    author_email='',
    url='',
    packages=find_packages('src'),
    package_dir={'': 'src'},
    scripts=[],
    setup_requires=['pytest-runner'],
    install_requires=[
        'jupyter==1.0.0',
        'torch==1.0.1.post2',
        'torchvision==0.2.2.post3',
        'matplotlib==3.0.3',
        'seaborn-0.9.0'
    ],
    tests_require=['pytest'],
    cmdclass={'test': PyTest},
    include_package_data=True,
    package_data={
        # Add all files under data/ directory.
        # This data will be part of this package.
        # Access them with pkg_resources module.
        # Folder with data HAVE TO be in some module, so dont add it to folder with tests, which SHOULD NOT be a module.
    },
)
