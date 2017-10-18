from setuptools import setup

# Get the long description from the README file
def readme():
    with open('README.rst') as f:
        return f.read()

setup(name='shgo',
      version='0.3.4',
      description='Simplicial homology global optimisation',
      url='https://github.com/stefan-endres/shgo',
      author='Stefan Endres',
      author_email='stefan.c.endres@gmail.com',
      license='MIT',
      packages=['shgo'],
      install_requires=[
          'scipy',
          'numpy'
      ],
      long_description=readme(),
      keywords='optimization',
      classifiers=[
          # How mature is this project? Common values are
          #   3 - Alpha
          #   4 - Beta
          #   5 - Production/Stable
          'Development Status :: 3 - Alpha',

          # Indicate who your project is intended for
          'Intended Audience :: Science/Research',
          'Intended Audience :: Developers',
          'Topic :: Scientific/Engineering',
          'Topic :: Scientific/Engineering :: Mathematics',

          # Pick your license as you wish (should match "license" above)
           'License :: OSI Approved :: MIT License',

          # Specify the Python versions you support here. In particular, ensure
          # that you indicate whether you support Python 2, Python 3 or both.
          'Programming Language :: Python :: 2.7',
          'Programming Language :: Python :: 3.5',
          'Programming Language :: Python :: 3.6',
      ],
      test_suite='shgo.shgo_tests',
      zip_safe=False)