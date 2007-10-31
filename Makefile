data/ABQ_Intersection:
	mkdir -p data
	sudo sshfs afraser@yks.lanl.gov:/n/projects/watcher/data data -o uid=1000 -o allow_other -o gid=1000
