
groundTruthTracks.txt: data/ABQ_Intersection
	cp data/ABQ_Intersection/groundTruthTracks.txt $@

data/ABQ_Intersection:
	mkdir -p data
	sudo sshfs afraser@yks.lanl.gov:/n/projects/watcher/data data -o uid=1000 -o allow_other -o gid=1000

ha.png: hackt.ckt
	convert hackt.ckt ha.png

model.pdf: model.tex ha.png
	pdflatex model.tex

# Local Variables:
# mode: makefile
# folded-file: t
# folding-internal-margins: 0
# End: