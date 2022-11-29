First 9.6e11 elements of series
```
AVX2 - 12 threads, Intel(R) Core(TM) i7-10850H
real    1m29.893s
user    17m25.471s
sys     0m0.258s

OMP_DISPLAY_ENV=TRUE OMP_NUM_THREADS=6 time ./harmonic_series
6 threads
real    1m30.69s
user    9m0.53s

2x AMD EPYC 9654 96-Core Processor
AVX-512
384 threds
real    0m1.817s
user    9m58.550s

OMP_DISPLAY_ENV=TRUE OMP_NUM_THREADS=192 time ./harmonic_series
192 threads
real    0m1.57s
user    6m58.85s
```
