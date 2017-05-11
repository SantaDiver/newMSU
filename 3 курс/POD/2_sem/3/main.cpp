#include <iostream>
#include <fstream>
#include <algorithm>
#include <vector>
#include <string>
#include <mpi.h>
#include <omp.h>
#include <complex>

using namespace std;

typedef complex<double> complexd;
typedef vector< vector <complexd > > matrix;
typedef unsigned long long ull;

int Rank = 0, procs_cnt;
double eps;
unsigned int my_seed = time(0);
int n, threads_cnt;


double MyRandom(unsigned seed)              { return (double) (rand_r(&seed)) / (RAND_MAX/100.0); }
int getBit(int a, int k)             { return (a & (1 << (n - k))) >> (n - k); }
int withBit(int & a, int k, int bit) {
    if (!bit) return a & (~(1 << (n - k)));
    return a | (bit << (n - k));
}

void Adamar(vector <complexd> & A, vector <complexd> & B, int k, matrix u) {
    int block_size = A.size();
    int block_shift = (1 << (n - k)) / block_size;

    if (block_shift == 0) {
        #pragma omp parallel for num_threads(threads_cnt)
        for (int i = 0; i < block_size; ++i) {
            int Ik = getBit(i, k);
            B[i] = u[Ik][0] * A[withBit(i, k, 0)] + u[Ik][1] * A[withBit(i, k, 1)];
        }
    }
    else {
        complexd * buff = new complexd[block_size];
        MPI_Sendrecv(
            A.data(), block_size, MPI::COMPLEX, Rank ^ block_shift, 0, buff,
            block_size, MPI::COMPLEX, Rank ^ block_shift, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE
        );

        #pragma omp parallel for num_threads(threads_cnt)
        for (int i = 0; i < block_size; ++i) {
            if (Rank < (Rank ^ block_shift))
                B[i] = u[0][0] * A[i] + u[0][1] * buff[i];
            else
                B[i] = u[1][0] * buff[i] + u[1][1] * A[i];
        }
        delete buff;
    }
}

double normal_dis_gen() {
    double S = 0.;
    for (int i = 0; i<12; ++i) { S += (double)rand_r(&my_seed)/RAND_MAX; }
    return S-6.;
}

complexd sc_mult(vector<complexd> & a, vector<complexd > & b) {
    complexd sum(0.0, 0.0);
    for (int i = 0; i < a.size(); ++i)
        sum += a[i] * conj(b[i]);
    return sum;
}

matrix mult(matrix & a, matrix & b) {
    matrix c(2, vector<complexd>(2, 0));
    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 2; ++j) {
            for (int k = 0; k < 2; ++k)
                c[i][j] += a[i][k] * b[k][j];
        }
    }
    return c;
}

void print(vector<complexd> v) {
    for(int i = 0 ; i < v.size(); ++i) cout << v[i];
    cout << endl;
}

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &procs_cnt);
    MPI_Comm_rank(MPI_COMM_WORLD, &Rank);

    n = atoi(argv[1]);
    threads_cnt = atoi(argv[2]);
    eps = atof(argv[3]);

    int vec_size = (1 << n);
    int block_size = vec_size / procs_cnt;
    vector<complexd > A_n(block_size), B_n(block_size), A_i(block_size), B_i(block_size);

    double mynormsum = 0.0, normsum;
    for (int i = 0; i < block_size; ++i) {
        A_n[i] = complexd(MyRandom(179 + Rank + time(0)),MyRandom(180 + Rank + time(0)));
        mynormsum += abs(A_n[i]) * abs(A_n[i]);
    }
    MPI_Allreduce(&mynormsum, &normsum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    normsum = sqrt(normsum);
    for (int i = 0; i < block_size; ++i)
        A_i[i] = A_n[i] = A_n[i] / normsum;

    MPI_Barrier(MPI_COMM_WORLD);
    double start = MPI_Wtime();

    matrix H(2, vector<complexd>(2));
    H[0][0] = 1 / sqrt(2); H[0][1] =  1 / sqrt(2);
    H[1][0] = 1 / sqrt(2); H[1][1] = -1 / sqrt(2);

    matrix U(2, vector<complexd>(2));

    double r = eps * normal_dis_gen();

    U[0][0] =  std::cos(r);U[0][1] =  std::sin(r);
    U[1][0] = -std::sin(r); U[1][1] =  std::cos(r);

    for (int i = 1; i <= n; ++i) {
        matrix Hn = mult(H, U);
        Adamar(A_i, B_i, i, H);
        A_i.swap(B_i);
        Adamar(A_n, B_n, i, Hn);
        A_n.swap(B_n);

    }

    double mytime = MPI_Wtime() - start, maxtime;
    MPI_Reduce(&mytime, &maxtime, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    complexd SUM = 0, mysum = sc_mult(B_n, B_i);
    MPI_Reduce(&mysum, &SUM, 1, MPI_DOUBLE_COMPLEX,MPI_SUM, 0, MPI_COMM_WORLD);

    if (Rank == 0) {
        double fidelity = abs(SUM) * abs(SUM);
        cout    << "procs: " << procs_cnt
                << ", cores: " << threads_cnt
                << ", n:" << n
                << ", e: " << eps
                << ", 1-fid: " << 1.0 - (float)fidelity
                << ", time: " << maxtime << endl;
    }

    MPI_Finalize();
    return 0;
}
