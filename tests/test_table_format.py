"""Tests for the shared Markdown table formatting."""

from __future__ import annotations

from synthea_quality.table_format import table_cell


def test_pipe_is_escaped():
    assert table_cell("a|b") == "a\\|b"


def test_newline_becomes_a_space():
    assert table_cell("a\nb") == "a b"


def test_crlf_becomes_one_space():
    assert table_cell("a\r\nb") == "a b"


def test_carriage_return_becomes_a_space():
    assert table_cell("a\rb") == "a b"


def test_without_specials_is_unchanged():
    assert table_cell("plain text 123") == "plain text 123"


def test_empty_is_unchanged():
    assert table_cell("") == ""


def test_unicode_is_unchanged():
    assert table_cell("Größe ≥ 5") == "Größe ≥ 5"
