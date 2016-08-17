from distutils.core import setup
from Cython.Build import cythonize
import numpy as np;

setup(
  name = "calimager",
  version="1.0",
  packages=["calimager"],
  include_package_data = True,
  scripts=["calimager/calimager.py"],
  description="Calibrated imager",
  url="https://github.com/transientskp/aartfaac-tools",
  author="Folkert Huizinga",
  author_email="f.huizinga@uva.nl",
  maintainer="Folkert Huizinga",
  maintainer_email="f.huizinga@uva.nl",
  requires=["matplotlib", "numpy"],
  py_modules=[],
  ext_modules=cythonize("calimager/imager.pyx"),
  include_dirs=[np.get_include()]
)
