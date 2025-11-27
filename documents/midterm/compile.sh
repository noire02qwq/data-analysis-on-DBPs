#!/bin/bash

# This script compiles the midterm_report.tex file.
# It runs pdflatex, then bibtex, and then pdflatex twice more
# to ensure all references and citations are correctly formatted.

FILENAME="midterm_report"

# Run pdflatex for the first time
pdflatex $FILENAME.tex

# Run bibtex to generate the bibliography
bibtex $FILENAME.aux

# Run pdflatex again to include the bibliography
pdflatex $FILENAME.tex

# Run pdflatex one more time to fix any cross-referencing issues
pdflatex $FILENAME.tex

echo "Compilation finished. Please check $FILENAME.pdf"
