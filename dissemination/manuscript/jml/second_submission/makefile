# command to call for compiling tex files
TEX = latexmk -pdf

# command to call for assembling separate PDFs into single PDF (e.g., the cover letter and the paper)
GS = gs -q -dNOPAUSE -dBATCH -sDEVICE=pdfwrite

# names of various files to be compiled
LETTER = Heal16implicit_letter
PAPER = Heal16implicit
SUP = sup_mat
FINAL = MKHealey.pdf

# rules for compiling
all:  paper letter
	$(GS) -sOutputFile=$(FINAL) $(LETTER).pdf $(PAPER).pdf
	#open $(PAPER).pdf

view: $(FINAL)
	open $(FINAL)

clean:
	rm -rfv log_files figures/*converted* *.fff *.ttt *.fls *.sum *.aux *.log *.bbl *.blg  *.out *.lot *.fdb_latexmk $(PAPER).pdf $(LETTER).pdf $(FINAL) missing_refs.txt

letter: $(LETTER).tex
	$(TEX) $(LETTER).tex

paper: $(PAPER).tex
	rm -f *.sum
	texcount -1 -sum $(PAPER).tex > wordcount.sum
	$(TEX) $(PAPER).tex
	$(TEX) $(SUP).tex