# -*- coding: utf-8 -*-

"""
Matchering
~~~~~~~~~~~~~~~~~~~

Audio Matching and Mastering Python Library

:copyright: (C) 2016-2022 Sergree
:license: GPLv3, see LICENSE for more details.

"""

__title__ = "matchering"

__author__ = "Sergree"
__credits__ = [
    "Sergey Grishakov",
    "Igor Isaev",
    "Chin Yun Yu",
    "Elizaveta Grishakova",
    "Zicklag",
]
__maintainer__ = "Sergree"
__email__ = "sergree@vk.com"
__license__ = "GPLv3"
__copyright__ = "Copyright (C) 2016-2022 Sergree"

__version__ = "2.0.6"

from .log.handlers import set_handlers as log
from .results import Result, pcm16, pcm24
from .defaults import Config
from .core import process
from .loader import load
from .checker import check
from .core import analyze, apply_analysis