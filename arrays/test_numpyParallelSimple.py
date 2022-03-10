from pathlib import Path

from testbook import testbook
import numpy as np

notebookPath = Path("arrays/numpyParallelSimple.ipynb")


@testbook(notebookPath.as_posix(), execute=range(5))
def test_func(tb):
    generateData = tb.get("generateData")
    xyLength = 3
    timesteps = 3
    output = generateData(xyLength=xyLength, timesteps=timesteps)
    assert isinstance(output, np.ndarray), f"type {type(output)}"
