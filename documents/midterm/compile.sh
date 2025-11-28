#!/bin/bash

# This script compiles the midterm_report.tex file.
# It runs pdflatex, then bibtex, and then pdflatex twice more
# to ensure all references and citations are correctly formatted.

FILENAME="midterm_report"

echo "Compiling $FILENAME.tex..."

# Run pdflatex for the first time
pdflatex -interaction=nonstopmode -file-line-error $FILENAME.tex
if [ $? -ne 0 ]; then
    echo "Error during first pdflatex run."
    exit 1
fi

# Run bibtex to generate the bibliography
bibtex $FILENAME.aux
if [ $? -ne 0 ]; then
    echo "Error during bibtex run."
    exit 1
fi

# Run pdflatex again to include the bibliography
pdflatex -interaction=nonstopmode -file-line-error $FILENAME.tex

# Run pdflatex one more time to fix any cross-referencing issues
pdflatex -interaction=nonstopmode -file-line-error $FILENAME.tex

echo "Compilation finished. Please check $FILENAME.pdf"
