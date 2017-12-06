CC = g++
CFLAGS = -fopenmp -std=c++11 -Wall
LDFLAGS = -larmadillo


final: StoIHT_test.o Sto_IHT.o Bayes_Sto_IHT.o Functions.o
	$(CC) -o final StoIHT_test.o Sto_IHT.o Bayes_Sto_IHT.o Functions.o $(LDFLAGS) $(CFLAGS)
StoIHT_test.o: StoIHT_test.cpp 
	$(CC) -c StoIHT_test.cpp $(CFLAGS)
Sto_IHT.o: Sto_IHT.cpp Sto_IHT.h
	$(CC) -c Sto_IHT.cpp $(CFLAGS)
Bayes_Sto_IHT.o: Bayes_Sto_IHT.cpp Bayes_Sto_IHT.h
	$(CC) -c Bayes_Sto_IHT.cpp $(CFLAGS)
Functions.o: Functions.cpp Functions.h
	$(CC) -c Functions.cpp $(CFLAGS)

clean:
	rm *.o
