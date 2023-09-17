# Copyright (c) OpenMMLab. All rights reserved.
from .backbones import *  # noqa
from .builder import (BACKBONES, HEADS, LOSSES, NECKS,
                      build_backbone, build_head, build_loss, build_Seg_model,
                      build_neck)

from .base_seg_model import BaseSEG
