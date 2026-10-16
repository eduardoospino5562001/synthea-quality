"""Tests for the shared Markdown table formatting."""

from __future__ import annotations

from synthea_quality.table_format import fixed, table_cell, trimmed


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


def test_fixed_shows_small_non_zero_values_with_two_significant_digits():
    assert fixed(0.001, 2) == "0.001"
    assert fixed(3.2e-05, 2) == "3.2e-05"
    assert fixed(-0.004, 2, sign=True) == "-0.004"
    assert fixed(0.004, 2, sign=True) == "+0.004"


def test_fixed_keeps_the_boundary_and_zero_as_before():
    assert fixed(0.005, 2) == "0.01"
    assert fixed(0.0049999, 2) == "0.005"
    assert fixed(0.0, 2) == "0.00"
    assert fixed(12.345, 2) == f"{12.345:.2f}"
    assert fixed(123456.789, 2) == f"{123456.789:.2f}"


def test_trimmed_shows_small_non_zero_values_with_two_significant_digits():
    assert trimmed(3.2e-05) == "3.2e-05"
    assert trimmed(-1e-07) == "-1e-07"
    assert trimmed(2e-10) == "2e-10"
    assert trimmed(4.9e-05) == "4.9e-05"


def test_trimmed_keeps_zero_and_the_boundary_as_before():
    assert trimmed(0.0) == "0"
    assert trimmed(-0.0) == "0"
    assert trimmed(5e-05) == "0.0001"
    assert trimmed(120.0) == "120"
    assert trimmed(1.015) == "1.015"
    assert trimmed(93.9) == "93.9"


def test_fixed_and_trimmed_match_the_previous_format_above_half_a_unit():
    values = [0.005, 0.01, 0.125, 1.0, 12.345, 93.9, 120.0, 1000.5, -2.5, -0.01]
    for value in values:
        assert fixed(value, 2) == f"{value:.2f}"
        assert fixed(value, 2, sign=True) == f"{value:+.2f}"
    for value in [0.00005, 0.0001, 0.0015, 1.015, 93.9, 120.0, -3.25]:
        expected = f"{value:.4f}".rstrip("0").rstrip(".")
        expected = "0" if expected == "-0" else expected
        assert trimmed(value) == expected
