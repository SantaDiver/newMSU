build: main.cpp
	mpic++ main.cpp -o main -fopenmp -m64

report: build
	chmod +x runner.sh
	./runner.sh

clean:
	rm -rf main *.o
