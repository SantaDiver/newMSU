#include <iostream>
#include <cmath>
#include <fstream>
#include <complex>
#include <algorithm>
#include <vector>
#include <string>
#include "mpi.h"
#include <omp.h>


using namespace std;
typedef complex<double> complexd;
typedef vector< vector <complexd > > matrix;
typedef unsigned long long ull;
uint threads_cnt;
int Rank = 0, n, procs_cnt;


double MyRandom(unsigned &seed)              { return (double) (rand_r(&seed)) / (RAND_MAX/100.0); }
int getBit(int a, int k)             { return (a & (1 << (n - k))) >> (n - k); }
int w1(int a, int k) { return a | (1 << (n - k)); }
int w0(int a, int k) { return a & (~(1 << (n - k))); }

int getIk(int i, int k1, int k2) {
    int p1 = n - k1, p2 = n - k2;
    return (((i & (1 << p1)) >> p1) << 1) + ((i & (1 << p2)) >> p2);
}

void doubleCubit(vector<complexd>& A, vector<complexd>& B, int k1, int k2, matrix u) {
    int block_size = A.size();
    int block_shift1 = (1 << (n - k1)) / block_size;
    int block_shift2 = (1 << (n - k2)) / block_size;
    if (block_shift1 == 0 && block_shift2 == 0) {
        #pragma omp parallel for num_threads(threads_cnt)
        for (int i = 0; i < block_size; ++i) {
            int ik = getIk(i, k1, k2);
            B[i] = u[ik][0] * A[w0(w0(i, k2), k1)] + u[ik][1] * A[w0(w1(i, k2), k1)] +
                   u[ik][2] * A[w1(w0(i, k2), k1)] + u[ik][3] * A[w1(w1(i, k2), k1)];
        }
    }
    else if (block_shift1 == 0 && block_shift2 != 0) {
        vector<complexd> tmp(block_size);

        MPI_Sendrecv(A.data(),block_size,MPI::COMPLEX,block_shift1^Rank,0,tmp.data(),
        block_size,MPI::COMPLEX,block_shift1^Rank,0,MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        #pragma omp parallel for num_threads(threads_cnt)
        for (int i = 0; i < block_size; ++i) {
            int ik = getIk(i, k1, k2);
            B[i] = u[ik][0] * A[w0(w0(i, k2), k1)] + u[ik][1] * tmp[w0(i, k1)] +
                   u[ik][2] * A[w1(w0(i, k2), k1)] + u[ik][3] * tmp[w1(i, k1)];
        }
    }
    else if (block_shift1 != 0 && block_shift2 == 0) {
        vector<complexd> tmp(block_size);

        MPI_Sendrecv(A.data(),block_size,MPI::COMPLEX,block_shift2^Rank,0,tmp.data(),
        block_size,MPI::COMPLEX,block_shift2^Rank,0,MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        #pragma omp parallel for num_threads(threads_cnt)
        for (int i = 0; i < block_size; ++i) {
            int ik = getIk(i, k1, k2);

            B[i] = u[ik][0] * A[w0(w0(i, k2), k1)] + u[ik][1] * A[w0(w1(i, k2), k1)] +
                   u[ik][2] * tmp[w0(i, k2)] + u[ik][3] * tmp[w1(i, k1)];

        }
    }
    else {
        vector<complexd> tmp1(block_size),tmp2(block_size),tmp3(block_size);

        MPI_Sendrecv(A.data(), block_size, MPI::COMPLEX, block_shift1^Rank, 0, tmp1.data(), block_size,
                     MPI::COMPLEX, block_shift1^Rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        MPI_Sendrecv(A.data(),block_size,MPI::COMPLEX,block_shift2^Rank,0,tmp2.data(), block_size,
                     MPI::COMPLEX,block_shift2^Rank,0,MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        MPI_Sendrecv(A.data(),block_size,MPI::COMPLEX,block_shift1^block_shift2^Rank,0,tmp3.data(), block_size,
                     MPI::COMPLEX,block_shift1^block_shift2^Rank,0,MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        #pragma omp parallel for num_threads(threads_cnt)
        for (int i = 0; i < block_size; ++i) {
            int ik = getIk(i, k1, k2);
            B[i] = u[ik][0] * A[w0(w0(i, k2), k1)] + u[ik][1] * tmp2[w0(i, k1)] +
                   u[ik][2] * tmp1[w0(i, k1)] + u[ik][3] * tmp3[i];

        }
    }
}

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &procs_cnt);
    MPI_Comm_rank(MPI_COMM_WORLD, &Rank);

    n = atoi(argv[1]);
    threads_cnt = atoi(argv[2]);
    uint k1 = atoi(argv[3]), k2 = atoi(argv[4]);

    uint vec_size = (1 << n);
    uint block_size = vec_size / procs_cnt;

    unsigned seed = time(0);


    vector<complexd > A(block_size), B(block_size);
    double mynormsum = 0.0, normsum = 0.0;
    for (uint i = 0; i < block_size; ++i) {
        A[i] = complexd(MyRandom(seed), MyRandom(seed));
        mynormsum += abs(A[i]) * abs(A[i]);
    }

    MPI_Allreduce(&mynormsum, &normsum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    normsum = sqrt(normsum);
    for (uint i = 0; i < block_size; ++i) A[i] /= normsum;

    MPI_Barrier(MPI_COMM_WORLD);

    double start = MPI_Wtime();
    matrix H(4, vector<complexd> (4, 0));
    H[0][0] = 1;
    H[1][1] = 1;
    H[2][3] = 1;
    H[3][2] = 1;

    doubleCubit(A, B, k1, k2, H);
    double mytime = MPI_Wtime() - start, maxtime;
    MPI_Reduce(&mytime, &maxtime, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    cerr.flush();
    if (Rank == 0) {
        cout    << "procs: "   << procs_cnt
                << ", cores: " << threads_cnt
                << ", n:"      << n
                << ", time: "  << maxtime << endl;
    }

    MPI_Finalize();
    return 0;
}
