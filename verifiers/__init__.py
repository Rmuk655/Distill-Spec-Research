# verifiers/__init__.py — make this folder importable as a package.
#
# The files inside (inference_util.py, node.py, util.py, verifier.py, khisti.py)
# are a verbatim copy of the reference implementation at
# https://github.com/Rmuk655/Distill-Spec-Research/tree/main/GBV
#
# Those files use plain imports like `from util import *` because they were
# written as a flat directory.  We add this folder to sys.path here so those
# imports still resolve when the parent scripts (train.py, eval.py,
# inference.py) import from `verifiers.*`.
import os, sys
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
