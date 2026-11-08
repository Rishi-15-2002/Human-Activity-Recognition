HAR DSF Project - Modular LaTeX Report

MAIN FILE
- main.tex is the master document. It imports each chapter from sections/.

FOLDER STRUCTURE
- sections/: separate LaTeX source file for each report section
- figures/: high-resolution plots generated from your uploaded dataset and notebook configuration
- screenshots/: add screenshots from the deployed Streamlit application
- main.tex: compile this file to generate the complete report

COMPILE
Recommended: Overleaf (upload the complete folder as a ZIP and set main.tex as the main document), or run:
  pdflatex main.tex
  pdflatex main.tex
Run pdflatex twice so the table of contents and figure/table references update.

REPLACE PLACEHOLDERS
Search the section files for [ ... ] and replace:
- institute/college and department
- student name and ID
- programme, semester, academic year
- guide name, city and report date
Add college_logo.png to the project root (or adjust the path in sections/01_title.tex).
Add application screenshots to screenshots/ using the filenames listed in screenshots/README.txt.

IMPORTANT CONTENT NOTES
- Reported model scores are reproduced by rerunning the supplied notebook's model settings on the supplied train/test files.
- The supplied feature documentation describes upstream sensor signal filtering and feature extraction. The submitted notebook consumes the precomputed 561-feature matrices; it does not re-run raw-sensor preprocessing.
- The Streamlit URL was supplied by the student. Live app behaviour was not independently inspected; add authentic screenshots and verify UI details before final submission.
- Dataset and evaluation results are not fabricated. Please verify all student/institute placeholders and references before submission.
