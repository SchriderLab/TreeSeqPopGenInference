all: ms flagelbagel

ms:
	tar -xzvf ms.tar.gz
	cd msdir; gcc -o ms ms.c streec.c rand1.c -lm
	mv msdir/ms .

flagelbagel:
	git clone https://github.com/Lswhiteh/pop_gen_cnn.git