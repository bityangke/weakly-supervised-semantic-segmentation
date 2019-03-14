auto_run:
	for sigma in 1 2 3 4 5 6 7 8 9 10 ; do \
		for lamda in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1 ; do \
			filename="sigma_"$$sigma"_lamda_"$$lamda"_result"; \
			python train.py --sigma $$sigma --lamda $$lamda >> display; \
			python test_miou.py  >> $$filename; \
		done \
	done
run:
	python train.py 
	python test_miou.py
