#include <iostream>
#include "mpi.h"
#include <cmath>
#include <complex>
#include <ctime>
#include <cstdlib>

using namespace std;

typedef complex<double> complexd;
typedef unsigned long long ull;

double MyRandom(unsigned seed)              { return (double) (rand_r(&seed)) / (RAND_MAX/100.0); }
ull getBit(ull a, ull k, ull n)             { return (a & (1 << (n - k))) >> (n - k); }
ull withBit(ull & a, ull k, ull bit, ull n) {
    if (!bit) return a & (~(1 << (n - k)));
    return a | (bit << (n - k));
}

int main(int argc, char* argv[]) {
    int rank = 0, threads_cnt;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &threads_cnt); MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    ull n = atoi(argv[1]), k = atoi(argv[2]);

    complexd u[2][2];
    u[0][0] = 1 / sqrt(2); u[0][1] =  1 / sqrt(2);
    u[1][0] = 1 / sqrt(2); u[1][1] = -1 / sqrt(2);

    if ((1 << n) % threads_cnt != 0) return -1;
    ull block_size = (1 << n) / threads_cnt;
    complexd *A = new complexd[block_size], *B = new complexd[block_size];

    double start = MPI_Wtime();
    for (ull i = 0; i < block_size; ++i)
        A[i] = complexd(MyRandom(179 + rank),MyRandom(180 + rank));

    ull block_shift = (1 << (n - k)) / block_size;

    if (block_shift == 0) {
        for (ull i = 0; i < block_size; ++i) {
            ull Ik = getBit(i, k, n);
            B[i] = u[Ik][0] * A[withBit(i, k, 0, n)] + u[Ik][1] * A[withBit(i, k, 1, n)];
        }
    }
    else {
        complexd * buff = new complexd[block_size];
        MPI_Sendrecv(
            A, block_size, MPI::COMPLEX, rank ^ block_shift, 0, buff,
            block_size, MPI::COMPLEX, rank ^ block_shift, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE
        );

        for (ull i = 0; i < block_size; ++i) {
            if (rank < (rank ^ block_shift))
                B[i] = u[0][0] * A[i] + u[0][1] * buff[i];
            else
                B[i] = u[1][0] * buff[i] + u[1][1] * A[i];
        }
    }
    double mytime = MPI_Wtime() - start, maxtime;
    MPI_Reduce(&mytime, &maxtime, 1, MPI_DOUBLE,MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0)
        cout << "k: " << k << ", qbits: " << n << ", procs: " << threads_cnt << ", time: " << maxtime << endl;

    MPI_Finalize();
    return 0;
}
