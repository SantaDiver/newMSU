build: main.cpp
	mpic++ -o main main.cpp

report: build
	chmod +x runner.sh
	./runner.sh

clean:
	rm -rf main *.o
