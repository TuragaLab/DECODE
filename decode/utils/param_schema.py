# this file specifies config / param schema for parsing using pydantic standard
from typing import Optional

from pydantic.dataclasses import dataclass
from pydantic.types import FilePath, DirectoryPath


@dataclass
class InOutSchemaTraining:
    calibration_file: FilePath
    experiment_out: DirectoryPath
    checkpoint_init: Optional[FilePath] = None
    model_init: Optional[FilePath] = None
