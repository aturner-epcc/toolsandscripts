/*
 * Basic BLAS benchmarks
 *
 * Originally written by A. Jackson, EPCC
 * ZGEMM added by A. Turner, EPCC
 */

#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <sys/time.h>
#include <string.h>
#include <getopt.h>
#include <strings.h>
#include <inttypes.h>
#include <complex.h>

#ifdef USE_CBLAS
    #include "cblas.h"
#endif

#define RUNS 20000


#ifdef INTEGER8
	#define Int long
#else
	#define Int int
#endif

typedef void (*benchmark_func_t) (Int n, Int runs, double *rtime, double *gflops);

void daxpy_(Int *N, double *alpha, double *x, Int *incx, double *Y, Int *incy);
void dgemm_(const char * TRANSA, const char *TRANSB, Int *m, Int *n, Int *k, double *alpha, double *A, Int *lda, double *B, Int *ldb, double *beta, double *C, Int *ldc);
void zgemm_(const char * TRANSA, const char *TRANSB, Int *m, Int *n, Int *k, double complex *alpha, double complex *A, Int *lda, double complex *B, Int *ldb, double complex *beta, double complex *C, Int *ldc);
void dgemv_(const char * TRANSA, Int *m, Int *n, double *alpha, double *A, Int *lda, double *B, Int *incb, double *beta, double *C, Int *incc);


int64_t calc_cycles(void);

int64_t calc_cycles(void)
{
    unsigned hi, lo;
    __asm__ __volatile__ ("rdtsc" : "=a"(lo), "=d"(hi));
    return (int64_t) (( (unsigned long long)lo)|( ((unsigned long long)hi)<<32 ));
}


double wtime()
{
	struct timeval tv;
	gettimeofday (&tv, NULL);
	return tv.tv_sec + tv.tv_usec / 1e6;
}

/*-----------------------------------------------------------------------------
 *  DGEMV Benchmark
 *-----------------------------------------------------------------------------*/
void benchmark_dgemv(char *order, Int n, Int m, Int Runs, double *rtime, double *gflops)
{
    Int i;
	double *A, *B, *C;
	double ts,te;
	double alpha=1, beta=1;
	double flops;
	Int incb = 1, incc = 1;

    A = malloc(sizeof(double) * n * m );
    if (strcasecmp(order, "T") == 0 ){
	   B = malloc(sizeof(double) * m );
	   C = malloc(sizeof(double) * n );
       for (i = 0; i < m; i++) {
		   B[i] = i*2+1;
       }
       for (i = 0; i < n; i++) {
		   C[i] = 1;
       }
    } else {
       B = malloc(sizeof(double) * n );
	   C = malloc(sizeof(double) * m );
       for (i = 0; i < n; i++) {
		   B[i] = i*2+1;
       }
       for (i = 0; i < m; i++) {
		   C[i] = 1;
       }
    }

	for ( i = 0; i < n; i++ ){
        for ( j = 0; j < m; j++ ){
            A[i][j] = i+j;
        }	
	}

    /*-----------------------------------------------------------------------------
     *  Warmup
     *-----------------------------------------------------------------------------*/
    dgemv_(&order, &n, &m, &alpha, A, &n, B,&incb, &beta, C, &incc);
    dgemv_(&order, &n, &m, &alpha, A, &n, B,&incb, &beta, C, &incc);
    dgemv_(&order, &n, &m, &alpha, A, &n, B,&incb, &beta, C, &incc);

    /*-----------------------------------------------------------------------------
     *  Benchmark
     *-----------------------------------------------------------------------------*/
    ts = wtime();
	for (i=0; i < Runs; i++){
		dgemv_(&order, &n, &m, &alpha, A, &n, B,&incb, &beta, C, &incc);
	}
	te = wtime();
	flops = 2.0 * n * m;
	flops /=1000*1000*1000;
	flops /= (te-ts)/Runs;

    *gflops = flops;
    *rtime = (te-ts)/Runs;

    free(A);
	free(B);
	free(C);

}


/*-----------------------------------------------------------------------------
 *  DGEMV Latency Test
 *-----------------------------------------------------------------------------*/
void benchmark_dgemv_latency(Int n, Int Runs, double *rtime, double *gflops)
{
    Int i;
	double *A, *B, *C;
	double ts,te;
	double alpha=1, beta=1;
	double flops;
	Int incb = 1, incc = 1;
    Int ld =1;
    uint64_t cy_start, cy_end, cy_sum;
    n = 1;

    A = malloc(sizeof(double) * n *n );
	B = malloc(sizeof(double) * n );
	C = malloc(sizeof(double) * n );

	for ( i = 0; i < n * n; i++){
		A[i]=i+1;
	}
	for (i = 0; i < n; i++) {
		B[i]=i*2+1;
		C[i]=1;
	}
    n = 0;
    /*-----------------------------------------------------------------------------
     *  Warmup
     *-----------------------------------------------------------------------------*/
    dgemv_("N", &n,&n,&alpha, A, &ld, B,&incb, &beta, C, &incc);
    dgemv_("N", &n,&n,&alpha, A, &ld, B,&incb, &beta, C, &incc);
    dgemv_("N", &n,&n,&alpha, A, &ld, B,&incb, &beta, C, &incc);

    /*-----------------------------------------------------------------------------
     *  Benchmark
     *-----------------------------------------------------------------------------*/
    cy_start = 0;
    cy_end = 0;
    cy_sum = 0;
    ts = wtime();
	for (i=0; i < Runs; i++){
        cy_start = calc_cycles();
		dgemv_("N", &n,&n,&alpha, A, &ld, B,&incb, &beta, C, &incc);
        cy_end = calc_cycles();
        cy_sum += (cy_end-cy_start);
	}
	te = wtime();
    cy_sum = cy_sum/Runs;
	flops = 2.0 * n *n;
	flops /=1000*1000*1000;
	flops /= (te-ts)/Runs;

    *gflops = flops;
    *rtime = (double) cy_sum;

    free(A);
	free(B);
	free(C);

}


/*-----------------------------------------------------------------------------
 *  DGEMM Benchmark
 *-----------------------------------------------------------------------------*/
void benchmark_dgemm(Int n, Int Runs, double *rtime, double *gflops)
{
    Int i;
	double *A, *B, *C;
	double ts,te;
	double alpha=1, beta=1;
	double flops;

    A = malloc(sizeof(double) * n *n );
	B = malloc(sizeof(double) * n *n );
	C = malloc(sizeof(double) * n *n );

	for ( i = 0; i < n * n; i++){
		A[i]=i+1;
		B[i]=i+0.5;
	}

	/*-----------------------------------------------------------------------------
	 *  Warmup
	 *-----------------------------------------------------------------------------*/
	dgemm_("N","N", &n,&n,&n,&alpha, A, &n, B,&n, &beta, C, &n);
	dgemm_("N","N", &n,&n,&n,&alpha, A, &n, B,&n, &beta, C, &n);
	dgemm_("N","N", &n,&n,&n,&alpha, A, &n, B,&n, &beta, C, &n);

	ts = wtime();
	for (i=0; i < Runs; i++){
		dgemm_("N","N", &n,&n,&n,&alpha, A, &n, B,&n, &beta, C, &n);
	}
	te = wtime();
	double h = (double) n / 1000.0;
	flops = 2.0 * h *h *h;
	flops /= ((te-ts)/Runs);
    *gflops = flops;
    *rtime = ((te-ts)/Runs);

    free(A);
	free(B);
	free(C);


}

/*-----------------------------------------------------------------------------
 *  DGEMM Benchmark (Latency)
 *-----------------------------------------------------------------------------*/
void benchmark_dgemm_latency(Int n, Int Runs, double *rtime, double *gflops)
{
    Int i;
	double *A, *B, *C;
	double ts,te;
	double alpha=1, beta=1;
	double flops;
    Int ld = 1;
    uint64_t cy_start, cy_end, cy_sum;

    n = 1;
    A = malloc(sizeof(double) * n *n );
	B = malloc(sizeof(double) * n *n );
	C = malloc(sizeof(double) * n *n );

	for ( i = 0; i < n * n; i++){
		A[i]=i+1;
		B[i]=i+0.5;
	}

    n = 0;
	/*-----------------------------------------------------------------------------
	 *  Warmup
	 *-----------------------------------------------------------------------------*/
	dgemm_("N","N", &n,&n,&n,&alpha, A, &ld, B,&ld, &beta, C, &ld);
	dgemm_("N","N", &n,&n,&n,&alpha, A, &ld, B,&ld, &beta, C, &ld);
	dgemm_("N","N", &n,&n,&n,&alpha, A, &ld, B,&ld, &beta, C, &ld);

    cy_sum = 0 ;
	ts = wtime();
	for (i=0; i < Runs; i++){
        cy_start = calc_cycles();
        dgemm_("N","N", &n,&n,&n,&alpha, A, &ld, B,&ld, &beta, C, &ld);
        cy_end = calc_cycles();
        cy_sum += (cy_end-cy_start);
	}
	te = wtime();
    cy_sum /= Runs;
	double h = (double) n / 1000.0;
	flops = 2.0 * h *h *h;
	flops /= ((te-ts)/Runs);
    *gflops = flops;
    *rtime = cy_sum;

    free(A);
	free(B);
	free(C);


}

/*-----------------------------------------------------------------------------
 *  ZGEMM Benchmark
 *-----------------------------------------------------------------------------*/
void benchmark_zgemm(Int n, Int Runs, double *rtime, double *gflops)
{
    Int i;
	double complex *A, *B, *C;
	double ts,te,ia,ib;
	double complex alpha=1, beta=1;
	double flops;

    A = malloc(sizeof(double complex) * n *n );
	B = malloc(sizeof(double complex) * n *n );
	C = malloc(sizeof(double complex) * n *n );

	for ( i = 0; i < n * n; i++){
		ia = i+1;
                ib = i+0.5;
		A[i]=ia + ia*I;
		B[i]=ib + ib*I;
	}

	/*-----------------------------------------------------------------------------
	 *  Warmup
	 *-----------------------------------------------------------------------------*/
	zgemm_("N","N", &n,&n,&n,&alpha, A, &n, B,&n, &beta, C, &n);
	zgemm_("N","N", &n,&n,&n,&alpha, A, &n, B,&n, &beta, C, &n);
	zgemm_("N","N", &n,&n,&n,&alpha, A, &n, B,&n, &beta, C, &n);

	ts = wtime();
	for (i=0; i < Runs; i++){
		zgemm_("N","N", &n,&n,&n,&alpha, A, &n, B,&n, &beta, C, &n);
	}
	te = wtime();
	double h = (double) n / 1000.0;
	flops = 2.0 * h *h *h;
	flops /= ((te-ts)/Runs);
    *gflops = flops;
    *rtime = ((te-ts)/Runs);

    free(A);
	free(B);
	free(C);


}

/*-----------------------------------------------------------------------------
 *  ZGEMM Benchmark (Latency)
 *-----------------------------------------------------------------------------*/
void benchmark_zgemm_latency(Int n, Int Runs, double *rtime, double *gflops)
{
    Int i;
	double complex *A, *B, *C;
	double ts,te, ia,ib;
	double complex alpha=1, beta=1;
	double flops;
    Int ld = 1;
    uint64_t cy_start, cy_end, cy_sum;

    n = 1;
    A = malloc(sizeof(double complex) * n *n );
	B = malloc(sizeof(double complex) * n *n );
	C = malloc(sizeof(double complex) * n *n );

	for ( i = 0; i < n * n; i++){
		ia = i+1;
                ib = i+0.5;
		A[i]=ia + ia*I;
		B[i]=ib + ib*I;
	}

    n = 0;
	/*-----------------------------------------------------------------------------
	 *  Warmup
	 *-----------------------------------------------------------------------------*/
	zgemm_("N","N", &n,&n,&n,&alpha, A, &ld, B,&ld, &beta, C, &ld);
	zgemm_("N","N", &n,&n,&n,&alpha, A, &ld, B,&ld, &beta, C, &ld);
	zgemm_("N","N", &n,&n,&n,&alpha, A, &ld, B,&ld, &beta, C, &ld);

    cy_sum = 0 ;
	ts = wtime();
	for (i=0; i < Runs; i++){
        cy_start = calc_cycles();
        zgemm_("N","N", &n,&n,&n,&alpha, A, &ld, B,&ld, &beta, C, &ld);
        cy_end = calc_cycles();
        cy_sum += (cy_end-cy_start);
	}
	te = wtime();
    cy_sum /= Runs;
	double h = (double) n / 1000.0;
	flops = 2.0 * h *h *h;
	flops /= ((te-ts)/Runs);
    *gflops = flops;
    *rtime = cy_sum;

    free(A);
	free(B);
	free(C);


}



/*-----------------------------------------------------------------------------
 *  Daxpy Benchmark
 *-----------------------------------------------------------------------------*/
void benchmark_daxpy(Int n, Int Runs, double *rtime, double *gflops)
{
	double *A, *B;
	double ts,te;
	double alpha=1;
	double flops;
	Int incx = 1, incy = 1;
    Int i;

    A = malloc(sizeof(double) * n );
	B = malloc(sizeof(double) * n );

    for ( i = 0; i < n ; i++){
		A[i]=i+1;
		B[i]=i+0.5;
	}
    /* Warm up */
    daxpy_(&n,&alpha, A, &incx, B, &incy);
	daxpy_(&n,&alpha, A, &incx, B, &incy);
	daxpy_(&n,&alpha, A, &incx, B, &incy);

    /*  Benchmark */
	ts = wtime();
	for (i=0; i < Runs; i++){
		daxpy_(&n,&alpha, A, &incx, B, &incy);
	}
	te = wtime();
	flops = 2.0 * n;
	flops /=1000.0*1000.0*1000.0;
	flops /= (te-ts)/Runs;
    *rtime = (te-ts)/Runs;
    *gflops = flops;
    free(A);
    free(B);
}

/*-----------------------------------------------------------------------------
 *  Daxpy Benchmark
 *-----------------------------------------------------------------------------*/
void benchmark_daxpy_latency(Int n, Int Runs, double *rtime, double *gflops)
{
	double *A, *B;
	double ts,te;
	double alpha=1;
	double flops;
	Int incx = 1, incy = 1;
    Int i;
    uint64_t cy_start, cy_end, cy_sum;

    A = malloc(sizeof(double) * n );
	B = malloc(sizeof(double) * n );

    for ( i = 0; i < n ; i++){
		A[i]=i+1;
		B[i]=i+0.5;
	}
    n = 0;
    /* Warm up */
    daxpy_(&n,&alpha, A, &incx, B, &incy);
	daxpy_(&n,&alpha, A, &incx, B, &incy);
	daxpy_(&n,&alpha, A, &incx, B, &incy);

    /*  Benchmark */
    cy_sum = 0;
	ts = wtime();
	for (i=0; i < Runs; i++){
        cy_start = calc_cycles();
		daxpy_(&n,&alpha, A, &incx, B, &incy);
        cy_end = calc_cycles();
        cy_sum += (cy_end-cy_start);

	}
	te = wtime();
    cy_sum /= Runs;
	flops = 2.0 * n;
	flops /=1000.0*1000.0*1000.0;
	flops /= (te-ts)/Runs;
    *rtime = cy_sum;
    *gflops = flops;
    free(A);
    free(B);
}



int main (int argc, char **argv) {
	Int n=-1, runs=-1;
    double rtime = 0, flops =0;
    int choice, skip=0, only=0;
    char *skip_str = NULL;
    char *only_str = NULL;
    char bk_name[128];
    char order[1];
    benchmark_func_t benchmark = NULL;
    int latency = 0;


    while (1)
    {
        static struct option long_options[] =
        {
            /* Use flags like so:
            {"verbose",    no_argument,    &verbose_flag, 'V'}*/
            /* Argument styles: no_argument, required_argument, optional_argument */
            {"help",    no_argument,    0,    'h'},
            {"runs",    required_argument, 0, 'r'},
            {"nsize",     required_argument, 0, 'n'},
            {"msize",     required_argument, 0, 'm'},
            {"trans",     required_argument, 0, 't'},
            {"benchmark", required_argument, 0, 'b'},
            {0,0,0,0}
        };
        int option_index = 0;

        /* Argument parameters:
            no_argument: " "
            required_argument: ":"
            optional_argument: "::" */
       choice = getopt_long( argc, argv, "hr:n:m:t:b:",
                    long_options, &option_index);


        if (choice == -1)
            break;

        switch( choice )
        {
            case 'h':
                {
                    printf("BLAS Benchmark");
                    printf("\n");
                    printf("Usage: %s [--help|-h] [--dim|-d N] [--runs|-r R] [--benchmark|-b NAME]\n", argv[0]);
                    printf("\n");
                    printf("The options are:\n");
                    printf(" [--help|-h]        Print this help.\n");
                    printf(" [--nsize|-n N]       Dimension, n, of the case.\n");
                    printf(" [--msize|-m N]       Dimension, m, of the case.\n");
                    printf(" [--trans|-t T or N]       Transpose (T) or normal (N) problem.\n");
                    printf(" [--runs|-r RUNS]   Number of runs to perform.\n");
                    printf(" [--benchmark|-b NAME] Name of the Benchmark\n");
                    printf("\n");
                    exit(0);

                }

                break;
            case 'n':
                n = atoi(optarg);
                break;
            case 'm':
                m = atoi(optarg);
                break;
            case 'r':
                runs = atoi(optarg);
                break;
            case 't':
                strncpy(order, optarg, 1);
                break;
            case 'b':
                if (strcasecmp(optarg, "?" ) == 0) {
                    printf("Possible Benchmarks are:\n");
                    printf(" - DAXPY               Benchmark the DAXPY operation\n");
                    printf(" - DGEMM               Benchmark the DGEMM operation\n");
                    printf(" - DGEMV               Benchmark the DGEMV operation\n");
                    printf(" - DAXPY_LATENCY       Benchmark the latency of the DAXPY operation\n");
                    printf(" - DGEMM_LATENCY       Benchmark the latency of the DGEMM operation\n");
                    printf(" - DGEMV_LATENCY       Benchmark the latency of the DGEMV operation\n");

                    return EXIT_FAILURE;
                }
                else if (strcasecmp(optarg, "DAXPY") == 0 ){
                    strncpy(bk_name, "DAXPY", 128);
                    benchmark = & benchmark_daxpy;
                }
                else if (strcasecmp(optarg, "DGEMM") == 0 ){
                    strncpy(bk_name, "DGEMM", 128);
                    benchmark = & benchmark_dgemm;
                }
                else if (strcasecmp(optarg, "ZGEMM") == 0 ){
                    strncpy(bk_name, "ZGEMM", 128);
                    benchmark = & benchmark_dgemm;
                }
                else if (strcasecmp(optarg, "DGEMV") == 0 ){
                    strncpy(bk_name, "DGEMV", 128);
                    benchmark = & benchmark_dgemv;
                }
                else if (strcasecmp(optarg, "DGEMV_LATENCY") == 0 ){
                    strncpy(bk_name, "DGEMV_LATENCY", 128);
                    benchmark = & benchmark_dgemv_latency;
                    latency = 1;
                }
                else if (strcasecmp(optarg, "DGEMM_LATENCY") == 0 ){
                    strncpy(bk_name, "DGEMM_LATENCY", 128);
                    benchmark = & benchmark_dgemm_latency;
                    latency = 1;
                }
                else if (strcasecmp(optarg, "DAXPY_LATENCY") == 0 ){
                    strncpy(bk_name, "DAXPY_LATENCY", 128);
                    benchmark = & benchmark_daxpy_latency;
                    latency = 1;
                }

                break;
            default:
                /* Not sure how to get here... */
                return EXIT_FAILURE;
        }
    }

    if ( n < 0 ) {
        printf("The dimension has to be set to a positive integer.\n");
        exit(1);
    }
    if ( runs < 0 ) {
        printf("The number of runs has to be set to a postive integer.\n");
        exit(1);
    }
    if ( skip && only ){
        printf("Either --skip or --only can be defined. Not both of them.\n");
        exit(1);
    }

    if (benchmark == NULL) {
        printf("No benchmark selected.\n");
        exit(1);
    }

    printf("# Dimension: %d\n", (int) n);
    printf("# Runs: %d \n", (int) runs);
    printf("# Benchmark: %s\n", bk_name);
    if (skip) printf("# Skip: %s\n", skip_str);
    if (only) printf("# Only: %s\n", only_str);
    if ( latency ) {
        printf("#%20s \n", "Latency (Cycles)");

    } else {
        printf("#%10s \t %10s\n", "Runtime", "GFlops");
    }

    /*-----------------------------------------------------------------------------
     *  Standalone Benchmark
     *-----------------------------------------------------------------------------*/
    benchmark(n, runs, &rtime, &flops);
    if ( latency ) {
        uint64_t urtime = (uint64_t) rtime;
        printf("%10" PRIu64 "\n", urtime);

    } else {
        printf("%10.8e \t %10.8e\n", rtime, flops);
    }
    if (skip_str) free(skip_str);
    if (only_str) free(only_str);
	return 0;
}


