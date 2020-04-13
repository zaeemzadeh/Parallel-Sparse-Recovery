CC = g++
CFLAGS = -fopenmp -std=c++11
LDFLAGS = -larmadillo

final: main.o Sto_IHT.o Bayes_Sto_IHT.o Functions.o Parallel_AMP.o
	$(CC) -o final *.o $(LDFLAGS) $(CFLAGS)
main.o: main.cpp 
	$(CC) -c main.cpp $(CFLAGS)
Sto_IHT.o: Sto_IHT.cpp Sto_IHT.h
	$(CC) -c Sto_IHT.cpp $(CFLAGS)
Bayes_Sto_IHT.o: Bayes_Sto_IHT.cpp Bayes_Sto_IHT.h
	$(CC) -c Bayes_Sto_IHT.cpp $(CFLAGS)
Parallel_AMP.o: Parallel_AMP.cpp Parallel_AMP.h
	$(CC) -c Parallel_AMP.cpp $(CFLAGS)
Functions.o: Functions.cpp Functions.h
	$(CC) -c Functions.cpp $(CFLAGS)

clean:
	rm *.o
	rm final
