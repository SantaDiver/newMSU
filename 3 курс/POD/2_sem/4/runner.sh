echo "" > output.txt;
K=0
for procs in 1 2 4
do
    for cores in 1 2 4
    do
        mpirun -np $procs ./main 24 $cores 1 2 >> output.txt
        printf "\n" >> output.txt
    done
    printf "\n" >> output.txt
done
