echo "" > output.txt;
K=0
for procs in 1
do
    for cores in 1
    do
        mpirun -np $procs ./main 20 $cores 0.01 >> output.txt
        printf "\n" >> output.txt
    done
    printf "\n" >> output.txt
done

# printf "||||||||||||||||||||||||||||||||||||||\n\n" >> output.txt
#
# for n in 24
# do
#     mpirun -np 4 ./main $n 8 0.01 >> output.txt
#     printf "\n" >> output.txt
# done
#
# printf "||||||||||||||||||||||||||||||||||||||\n\n" >> output.txt
#
#
# for e in 0.1
# do
#     mpirun -np 4 ./main 26 8 $e >> output.txt
#     printf "\n" >> output.txt
# done
