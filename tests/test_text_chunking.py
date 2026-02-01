# coding=utf-8
# SPDX-License-Identifier: Apache-2.0
"""
Tests for text chunking functionality.
"""

import pytest
from fastapi.testclient import TestClient

from api.services.text_processing import split_into_chunks


class TestSplitIntoChunks:
    """Tests for split_into_chunks()."""

    def test_empty_string_returns_empty_list(self):
        assert split_into_chunks("") == []

    def test_whitespace_only_returns_empty_list(self):
        assert split_into_chunks("   ") == []

    def test_short_text_returns_single_chunk(self):
        text = "Hello world."
        result = split_into_chunks(text)
        assert result == [text]

    def test_text_at_exact_threshold_returns_single_chunk(self):
        text = "A" * 500
        result = split_into_chunks(text, max_chunk_size=500)
        assert result == [text]

    def test_splits_on_period(self):
        s1 = "A" * 300 + "."
        s2 = "B" * 300 + "."
        text = f"{s1} {s2}"
        result = split_into_chunks(text, max_chunk_size=400)
        assert len(result) == 2
        assert result[0] == s1
        assert result[1] == s2

    def test_splits_on_exclamation(self):
        s1 = "A" * 300 + "!"
        s2 = "B" * 300 + "!"
        text = f"{s1} {s2}"
        result = split_into_chunks(text, max_chunk_size=400)
        assert len(result) == 2

    def test_splits_on_question_mark(self):
        s1 = "A" * 300 + "?"
        s2 = "B" * 300 + "?"
        text = f"{s1} {s2}"
        result = split_into_chunks(text, max_chunk_size=400)
        assert len(result) == 2

    def test_splits_on_semicolon(self):
        s1 = "A" * 300 + ";"
        s2 = "B" * 300 + ";"
        text = f"{s1} {s2}"
        result = split_into_chunks(text, max_chunk_size=400)
        assert len(result) == 2

    def test_splits_on_paragraph_break(self):
        s1 = "A" * 300
        s2 = "B" * 300
        text = f"{s1}\n\n{s2}"
        result = split_into_chunks(text, max_chunk_size=400)
        assert len(result) == 2
        assert result[0] == s1
        assert result[1] == s2

    def test_oversized_single_sentence_kept_whole(self):
        text = "A" * 800
        result = split_into_chunks(text, max_chunk_size=500)
        assert len(result) == 1
        assert result[0] == text

    def test_packs_multiple_short_sentences_into_one_chunk(self):
        sentences = "First. Second. Third."
        result = split_into_chunks(sentences, max_chunk_size=500)
        assert len(result) == 1
        assert result[0] == sentences

    def test_custom_max_chunk_size(self):
        text = "Short. Also short."
        result = split_into_chunks(text, max_chunk_size=10)
        assert len(result) == 2
        assert result[0] == "Short."
        assert result[1] == "Also short."

    def test_natural_text_splits_at_sentences(self):
        text = (
            "The quick brown fox jumped over the lazy dog. "
            "It was a beautiful day in the neighborhood. "
            "The sun was shining and the birds were singing. "
            "Everyone was happy and content with the weather."
        )
        result = split_into_chunks(text, max_chunk_size=100)
        assert len(result) > 1
        # Each chunk should end with a period (sentence boundary)
        for chunk in result[:-1]:
            assert chunk.rstrip().endswith(".")

    def test_preserves_all_text(self):
        """Verify no text is lost during chunking."""
        text = (
            "Sentence one. Sentence two! Sentence three? "
            "Sentence four; Sentence five."
        )
        result = split_into_chunks(text, max_chunk_size=30)
        rejoined = " ".join(result)
        # All original words should be present
        for word in ["one", "two", "three", "four", "five"]:
            assert word in rejoined


class TestSchemaAcceptsLongInput:
    """Test that the schema no longer rejects text over 4096 chars."""

    def test_speech_request_accepts_long_input(self):
        from api.structures.schemas import OpenAISpeechRequest

        long_text = "Hello world. " * 500  # ~6500 chars
        req = OpenAISpeechRequest(input=long_text)
        assert len(req.input) > 4096

    def test_voice_clone_request_accepts_long_input(self):
        from api.structures.schemas import VoiceCloneRequest

        long_text = "Hello world. " * 500
        req = VoiceCloneRequest(input=long_text, ref_audio="dGVzdA==")
        assert len(req.input) > 4096
