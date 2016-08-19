from distutils.core import setup

setup(
  name = "calimager",
  version="1.0",
  packages=[],
  scripts=["calimager.py"],
  description="Calibrated imager",
  url="https://github.com/transientskp/aartfaac-calimager",
  author="Folkert Huizinga",
  author_email="f.huizinga@uva.nl",
  maintainer="Folkert Huizinga",
  maintainer_email="f.huizinga@uva.nl",
  requires=["matplotlib", "numpy", "gfft"]
)
