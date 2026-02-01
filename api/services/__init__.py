# coding=utf-8
# SPDX-License-Identifier: Apache-2.0
"""
Services package for the TTS API.
"""

from .text_processing import normalize_text, split_into_chunks, NormalizationOptions

__all__ = [
    "normalize_text",
    "split_into_chunks",
    "NormalizationOptions",
]
