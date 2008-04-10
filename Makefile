distribution: model.pdf
	mkdir -p distribution
	for file in SeqKeys.el View.py ha* mvx.py util.py model.pdf model.tex README demo.py; do cp $$file distribution/$$file; done
ABQ_prof: mvx.py util.py ABQ_track.py
	python -m cProfile -o $@ ABQ_track.py --track
groundTruthTracks.txt: data/ABQ_Intersection
	cp data/ABQ_Intersection/abqGroundTruth.trx $@

AMF_tracks0.txt AMF_tracks1.txt AMF_tracks2.txt:
	python ABQ_track.py --track

AMF_accel.txt: ABQ_track.py
	python ABQ_track.py --accel

data/ABQ_Intersection:
	mkdir -p data
	sudo sshfs afraser@yks.lanl.gov:/n/projects/watcher/data data -o uid=1000 -o allow_other -o gid=1000

ha.png: hackt.ckt
	convert hackt.ckt ha.png

%.pdf: %.ckt
	epstopdf $*.ckt --outfile=$@

model.pdf: model.tex ha.pdf ha2.pdf
	pdflatex model.tex

%.pdf: %.tex
	pdflatex $<
# Local Variables:
# mode: makefile
# folded-file: t
# folding-internal-margins: 0
# End: