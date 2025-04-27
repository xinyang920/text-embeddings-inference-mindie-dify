# Copyright Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.

import os
from dataclasses import dataclass
from loguru import logger


@dataclass
class EnvVar:
    """
    Reads environment variables to configure the embedding or reranker model backend and device ID for use in
    other components of TEI-MindIE.

    Attributes:
        backend: The backend model type to be used for inference (e.g., 'atb' or 'mindietorch').
        device_id: The device id to load model on (between '0' and '7').

    Raises:
        ValueError: If the backend does not belong to ['atb', 'mindietorch'] or the device_id is invalid.
    """

    backend: str = os.getenv("TEI_NPU_BACKEND", 'mindietorch')
    device_id: str = os.getenv("TEI_NPU_DEVICE", '0') 

    def __post_init__(self):
        logger.info(self.dict())
        if self.backend not in ['atb', 'mindietorch']:
            raise ValueError("Your model backend is invalid.")
        if not self.device_id.isdigit():
            raise ValueError("Your device_id is invalid.")
        if int(self.device_id) < 0:
            raise ValueError("Your device_id is invalid.")
        if int(self.device_id) >= 8:
            logger.warning(f'Your device_id is {self.device_id}.')

    def dict(self):
        return self.__dict__

ENV = EnvVar()