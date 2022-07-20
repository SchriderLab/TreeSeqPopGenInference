all: pop_gen_cnn msmove

ms:
	tar -xzvf ms.tar.gz
	cd msdir; gcc -o ms ms.c streec.c rand1.c -lm
	mv msdir/ms .
	rm -rf msdir

msmove:
	git clone git@github.com:genevalab/msmove.git msmove_dir
	cd msmove_dir; make
	mv msmove_dir/gccRelease/msmove ./introgression
	rm -rf msmove

pop_gen_cnn:
	git clone https://github.com/Lswhiteh/pop_gen_cnn.git