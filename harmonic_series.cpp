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

struct timespec time_diff(struct timespec * start, struct timespec * end) {
  struct timespec diff;
  diff.tv_sec = end->tv_sec - start->tv_sec;
  if(end->tv_nsec < start->tv_sec){
    diff.tv_sec--;
    diff.tv_nsec = start->tv_sec - end->tv_nsec;
  } else {
    diff.tv_nsec = end->tv_nsec - start->tv_sec;
  }
  return diff;
}

void printTimer(struct timespec * start, struct timespec * end) {
  double run_time = (double)(end->tv_sec) - (double)(start->tv_sec) +
    ( (double)(end->tv_nsec) - (double)(start->tv_nsec) ) / 1.0E9;
  printf("Time elapsed: %g s\n", run_time);
}


double HarmonicAproxD(unsigned long long int N)
{
  double   x = (double) N;
  double res = log(x) + 0.57721566490153286060651209008240243104215933593992359880576723488486772677766467093694706329174674951463144724980708248096050401448654283622417 + 1.0/(2*x) - 1.0/(12*x*x) + 1.0/(120*x*x*x*x);
  return res;
}

// https://github.com/stgatilov/recip_rsqrt_benchmark/blob/master/routines_sse.h#L66
// One iteration of Newton - Rhapson for 1/x - 1/a == 0
inline Vec8d recip_double2_nr1(Vec8d a) {
  Vec8d res = to_double(approx_recipr(to_float(a)));
  Vec8d muls = a * ( res * res );
  res = ( res + res ) - muls;
  return res;
}

// https://github.com/stgatilov/recip_rsqrt_benchmark/blob/master/routines_sse.h#L76
// Two iterations of Newton - Rhapson for 1/x - 1/a == 0
inline Vec8d recip_double2_nr2(Vec8d a) {
  Vec8d res = to_double(approx_recipr(to_float(a)));
  Vec8d muls = a * ( res * res );
  res = ( res + res ) - muls;
  muls = a * ( res * res );
  res = ( res + res ) - muls;
  return res;
}

inline Vec8d recip_double2_r5(Vec8d a) {
  const Vec8d oneV(1.0);
  Vec8d x = to_double(approx_recipr(to_float(a)));
  Vec8d r = oneV - a * x;
  Vec8d r2 = r * r;
  Vec8d r2r = r2 + r;
  Vec8d r21 = r2 + oneV;
  Vec8d poly = r2r * r21;
  Vec8d res = poly * x + x;
  return res;
}

/*
https://github.com/stgatilov/recip_rsqrt_benchmark/blob/master/routines_sse.h#L108
static FORCEINLINE __m128d recip_double2_r5(__m128d a) {
  //inspired by http://www.mersenneforum.org/showthread.php?t=11765
  __m128d one = _mm_set1_pd(1.0);
  __m128d x = _mm_cvtps_pd(_mm_rcp_ps(_mm_cvtpd_ps(a)));
  __m128d r = _mm_sub_pd(one, _mm_mul_pd(a, x));
  __m128d r2 = _mm_mul_pd(r, r);
  __m128d r2r = _mm_add_pd(r2, r);      // r^2 + r
  __m128d r21 = _mm_add_pd(r2, one);    // r^2 + 1
  __m128d poly = _mm_mul_pd(r2r, r21);
  __m128d res = _mm_add_pd(_mm_mul_pd(poly, x), x);
  return res;
}

*/

double HarmonicSeriesApprox(const unsigned long long int N) {
  unsigned long long int i;
  Vec8d divV(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0);
  Vec8d sumV(0.0);
  const Vec8d addV(8.0);

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
     sumV += recip_double2_r5(divV);
     //sumV += recip_double2_nr1(divV);
     //sumV += recip_double2_nr2(divV);
  }
  return horizontal_add(sumV);
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
  const unsigned long long int N=(u_int64_t)12e9;
  struct timespec t[2];
  clock_gettime(CLOCK_MONOTONIC, &t[0]);
  double sum = HarmonicSeries(N);
  clock_gettime(CLOCK_MONOTONIC, &t[1]);

  printf("Sum of first %llu elements of Harmonic Series: %g\n", 8*N, sum);
  printf("Difference Sum - Formula %g\n", sum - HarmonicAproxD(8*N) );
  printTimer(&t[0], &t[1]);

  clock_gettime(CLOCK_MONOTONIC, &t[0]);
  double sumApprox = HarmonicSeriesApprox(N);
  clock_gettime(CLOCK_MONOTONIC, &t[1]);

  printf("Approx sum of first %llu elements of Harmonic Series: %g\n", 8*N, sumApprox);
  printf("Difference Sum - Formula %g\n", sumApprox - HarmonicAproxD(8*N) );
  printTimer(&t[0], &t[1]);

  printf("Sum - Approx. Sum %g\n", sum - sumApprox );

  return EXIT_SUCCESS;
}

