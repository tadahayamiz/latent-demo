# -*- coding: utf-8 -*-
"""
Created on Wed 13 15:46:32 2026

Small display helpers for notebooks.

@author: tadahaya
"""
from __future__ import annotations


def display_markdown(text: str) -> None:
    """Display Markdown in notebooks, falling back to plain text."""
    try:
        from IPython.display import Markdown, display

        display(Markdown(text))
    except Exception:
        print(text)


def display_dataframe(df, max_rows: int = 10) -> None:
    """Display a DataFrame in notebooks, falling back to print."""
    try:
        from IPython.display import display

        display(df.head(max_rows) if hasattr(df, "head") else df)
    except Exception:
        print(df.head(max_rows) if hasattr(df, "head") else df)


def show_question(text: str) -> None:
    display_markdown(f"> **問い**  \\n> {text}")
