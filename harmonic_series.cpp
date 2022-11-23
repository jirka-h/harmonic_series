/*
g++ -Wall -Wextra -O3 -g                  -I include -fopenmp -m64 -mavx2 -mfma                                     -std=c++17 -o harmonic_series harmonic_series.cpp
g++ -Wall -Wextra -g -fsanitize=undefined -I include -fopenmp -m64 -mavx2 -mfma                                     -std=c++17 -o harmonic_series harmonic_series.cpp
g++ -Wall -Wextra -O3 -g                  -I include -fopenmp -m64 -mavx512f -mfma -mavx512vl -mavx512bw -mavx512dq -std=c++17 -o harmonic_series harmonic_series.cpp
g++ -Wall -Wextra -g -fsanitize=undefined -I include -fopenmp -m64 -mavx512f -mfma -mavx512vl -mavx512bw -mavx512dq -std=c++17 -o harmonic_series harmonic_series.cpp

https://stackoverflow.com/questions/74527011/how-to-use-vector-class-library-for-avx-vectorization-together-with-the-openmp
*/

#include <iostream>
#include <vectorclass.h>
#include <omp.h>
#include <csignal>
#include <vector>
#include <cmath>

using std::cout;
using std::cin;
using std::endl;

// https://stackoverflow.com/questions/8138168/signal-handling-in-openmp-parallel-program
class Unterminable {
    sigset_t oldmask, newmask;
    std::vector<int> signals;

public:
    Unterminable(std::vector<int> signals) : signals(signals) {
        sigemptyset(&newmask);
        for (int signal : signals)
            sigaddset(&newmask, signal);
        sigprocmask(SIG_BLOCK, &newmask, &oldmask);
    }

    Unterminable() : Unterminable({SIGINT, SIGTERM}) {}

    int poll() {
        sigset_t sigpend;
        sigpending(&sigpend);
        for (int signal : signals) {
            if (sigismember(&sigpend, signal)) {
                int sigret;
                if (sigwait(&newmask, &sigret) == 0)
                    return sigret;
                break;
            }
        }
        return -1;
    }

    ~Unterminable() {
        sigprocmask(SIG_SETMASK, &oldmask, NULL);
    }
};

static __inline__ u_int64_t start_clock() {
    // See: Intel Doc #324264, "How to Benchmark Code Execution Times on Intel...",
    u_int32_t hi, lo;
    __asm__ __volatile__ (
        "CPUID\n\t"
        "RDTSC\n\t"
        "mov %%edx, %0\n\t"
        "mov %%eax, %1\n\t": "=r" (hi), "=r" (lo)::
        "%rax", "%rbx", "%rcx", "%rdx");
    return ( (u_int64_t)lo) | ( ((u_int64_t)hi) << 32);
}

static __inline__ u_int64_t stop_clock() {
    // See: Intel Doc #324264, "How to Benchmark Code Execution Times on Intel...",
    u_int32_t hi, lo;
    __asm__ __volatile__(
        "RDTSCP\n\t"
        "mov %%edx, %0\n\t"
        "mov %%eax, %1\n\t"
        "CPUID\n\t": "=r" (hi), "=r" (lo)::
        "%rax", "%rbx", "%rcx", "%rdx");
    return ( (u_int64_t)lo) | ( ((u_int64_t)hi) << 32);
}

double HarmonicAproxD(unsigned long long int N)
{
  double   x = (double) N;
  double res = log(x) + 0.57721566490153286060651209008240243104215933593992359880576723488486772677766467093694706329174674951463144724980708248096050401448654283622417 + 1.0/(2*x) - 1.0/(12*x*x) + 1.0/(120*x*x*x*x);
  return res;
}

double HarmonicSeries(const unsigned long long int N) {
  unsigned long long int i;
  Vec8d divV(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0);
  Vec8d sumV(0.0);
  const Vec8d addV(8.0);
  const Vec8d oneV(1.0);

#if 1
  const Vec8d startdivV = divV;
  bool first_loop = true;
  #pragma omp declare reduction( + : Vec8d : omp_out = omp_out + omp_in ) initializer (omp_priv=omp_orig)
//It's important to mark "first_loop" variable as firstprivate so that each private copy gets initialized.
  #pragma omp parallel for firstprivate(first_loop) lastprivate(divV) reduction(+:sumV)
  for(i=0; i<N; ++i) {
    if (first_loop) {
      divV = startdivV + i * addV;
      first_loop = false;
    } else {
      divV += addV;
    }
    sumV += oneV / divV;
  }
#if 0
  for (int j=0; j<8; ++j) {
    printf("%.0f, ", divV[j]);
  }
  printf("\n");
#endif

#else
/*
Manual reduction
This code blindly assumes that N is divisible by number of threads
https://stackoverflow.com/questions/18746282/openmp-schedulestatic-with-no-chunk-size-specified-chunk-size-and-order-of-as
*/
  unsigned long long int start, end;
  int thread_num, num_threads;
  Vec8d localsumV;
  Vec8d localdivV;

  #pragma omp parallel private( i, thread_num, num_threads, start, end, localsumV, localdivV )
  {
    thread_num = omp_get_thread_num();
    num_threads = omp_get_num_threads();
    start = thread_num * N / num_threads;
    end = (thread_num + 1) * N / num_threads;
    localsumV = sumV;
    localdivV = start* addV + divV;
    for(i=start; i<end; ++i) {
      localsumV += oneV / localdivV;
      localdivV += addV;
    }
    #pragma omp critical
    {
      sumV += localsumV;
#if 0
      printf("Please note - numbers will be bigger by one than the last value used for the summation\n");
      printf("Thread %d:\t", omp_get_thread_num());
      for (int j=0; j<8; ++j) {
        printf("%.0f, ", localdivV[j]);
      }
      printf("\n");
#endif
    }
  }
#endif
  return horizontal_add(sumV);
}


int main(int argc, char** argv) {
/*
  if (argc != 2) {
    fprintf(stderr, "Program needs exactly one argument - input file!\n");
    return EXIT_FAILURE;
  }
*/
  const unsigned long long int N=(u_int64_t)12e10;
  struct timespec t[2];
  u_int64_t rdtsc[2];
  clock_gettime(CLOCK_MONOTONIC, &t[0]);
  rdtsc[0] = start_clock();
  double sum = HarmonicSeries(N);
  rdtsc[1] = start_clock();
  clock_gettime(CLOCK_MONOTONIC, &t[1]);

  printf("Sum of first %llu elements of Harmonic Series: %g\n", 8*N, sum);
  printf("Difference Sum - Formula %g\n", sum - HarmonicAproxD(8*N) );


  return EXIT_SUCCESS;
}

