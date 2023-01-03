/*
g++ -Wall -Wextra -O3 -g                  -I include -fopenmp -m64 -mavx2 -mfma                                     -std=c++17 -o harmonic_series harmonic_series.cpp

g++ -Wall -Wextra -O3 -g                  -I include -fopenmp -m64 -mavx2 -mfma                                     -std=c++17 -o harmonic_series harmonic_series.cpp
g++ -Wall -Wextra -g -fsanitize=undefined -I include -fopenmp -m64 -mavx2 -mfma                                     -std=c++17 -o harmonic_series harmonic_series.cpp
g++ -Wall -Wextra -O3 -g                  -I include -fopenmp -m64 -mavx512f -mfma -mavx512vl -mavx512bw -mavx512dq -std=c++17 -o harmonic_series harmonic_series.cpp
g++ -Wall -Wextra -g -fsanitize=undefined -I include -fopenmp -m64 -mavx512f -mfma -mavx512vl -mavx512bw -mavx512dq -std=c++17 -o harmonic_series harmonic_series.cpp

https://stackoverflow.com/questions/74527011/how-to-use-vector-class-library-for-avx-vectorization-together-with-the-openmp
*/
#include <iostream>
#include <vectorclass.h>
#include <omp.h>
#include <cmath>

using std::cout;
using std::cin;
using std::endl;

double time_diff(const struct timespec * start, const struct timespec * end) {
  double run_time = (double)(end->tv_sec) - (double)(start->tv_sec) +
    ( (double)(end->tv_nsec) - (double)(start->tv_nsec) ) / 1.0E9;
  return run_time;
}

void printTimer(const struct timespec * start, const struct timespec * end) {
  double run_time = time_diff(start, end);
  printf("Time elapsed: %g s\n", run_time);
}

int string_to_double(const char *a, double *r, const double min, const double max) {
  char *p;
  double d = strtod(a, &p);
  if ((p == a) || (*p != 0) || errno == ERANGE || (d < min ) || (d > max ) ) {
    fprintf(stderr,"ERROR when parsing \"%s\" as double value. Expecting number in range < %g - %g > in double notation, see \"man strtod\" for details.", a, min, max);
    return 1;
  }
  *r = d;
  return 0;
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
  if (argc != 2) {
    fprintf(stderr, "Program needs exactly one argument - runtime in seconds!\n");
    return EXIT_FAILURE;
  }

  double runtime;
  if ( string_to_double(argv[1], &runtime, 2, 31536000) > 0 ) return EXIT_FAILURE;

  unsigned long long int N = (u_int64_t)1e8;
  struct timespec t[2];
  double sum;
  double elapsed_time;

  //Get number of iterations
  clock_gettime(CLOCK_MONOTONIC, &t[0]);
  sum = HarmonicSeries(N);
  clock_gettime(CLOCK_MONOTONIC, &t[1]);
  elapsed_time = time_diff(&t[0], &t[1]);

  if (elapsed_time < runtime) {
    printf("Estimating number of terms of sum to reach runtime %g seconds.\n", runtime);
    double terms = ( (double) N / elapsed_time * runtime );
    //round to 2 valid digits
    double exp = floor(log10(terms));
    if (exp > 2 ) {
      terms = round(terms/pow(10.0, exp-1)) * pow(10.0, exp-1);
    }
    printf("%g terms took %g seconds. Need %g terms.\n", (double)(8*N), elapsed_time, 8*terms);
    N = (unsigned long long int) terms;

    clock_gettime(CLOCK_MONOTONIC, &t[0]);
    sum = HarmonicSeries(N);
    clock_gettime(CLOCK_MONOTONIC, &t[1]);
    elapsed_time = time_diff(&t[0], &t[1]);
  }

  printf("Sum of first %llu elements of Harmonic Series: %g completed in %g seconds.\n", 8*N, sum, elapsed_time);
  printf("Difference Sum - Formula %g\n", sum - HarmonicAproxD(8*N) );
  printTimer(&t[0], &t[1]);
  printf("Avg: %g operations/second\n", (double) (8 * N) / elapsed_time);

  return EXIT_SUCCESS;
}

