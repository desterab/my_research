# command to call for compiling tex files
TEX = latexmk -pdf

# command to call for assembling separate PDFs into single PDF (e.g., the cover letter and the paper)
GS = gs -q -dNOPAUSE -dBATCH -sDEVICE=pdfwrite

# names of various files to be compiled
LETTER = Heal16implicit_letter
PAPER = Heal16implicit
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
	texcount -1 -sum $(PAPER).tex > all.sum # total word count all sections
	texcount -sub=section $(PAPER).tex  | grep "Method" | sed -e 's/+.*//' > temp.sum # methods word count
	texcount -sub=section $(PAPER).tex  | grep "Results" | sed -e 's/+.*//' >> temp.sum # results word count
	paste -s -d"+" temp.sum | bc >> all.sum # sum methods and results, add to all.sum
	paste -s -d"-" all.sum | bc > wordcount.sum # subtract lines of all to get total excluding methods and results
	$(TEX) $(PAPER).tex