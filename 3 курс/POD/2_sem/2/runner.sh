echo "" > output.txt;
K=0
for k in 1 4 -1
do
    for n in 25 26
    do
        if [ $k = -1 ]
        then
            K=$n
        else
            K=$k
        fi
        for procs in 1 2 4
        do
            mpirun -np $procs ./main $n $K
        done
        printf "\n" >> output.txt
    done
    printf "||||||||||||||||||||||||||||||||||||||\n\n" >> output.txt
done
