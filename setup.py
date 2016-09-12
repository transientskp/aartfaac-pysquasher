from distutils.core import setup

setup(
  name = "pysquasher",
  version="1.0",
  packages=[],
  scripts=["pysquasher.py", "rms.py"],
  description="Squash calibrated images",
  url="https://github.com/transientskp/aartfaac-pysquasher",
  author="Folkert Huizinga",
  author_email="f.huizinga@uva.nl",
  maintainer="Folkert Huizinga",
  maintainer_email="f.huizinga@uva.nl",
  requires=["astropy", "numpy", "gfft"]
)
