#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "sys/time.h"
#include "ising.h"

#define DefNumPD 517	// Default Number of Points per Dimention
#define DefNumI 10		// Default Number of Iterations
#define DefExp 0		// Default Value to Export Data (0 = False)

enum Data_Types { CHAR_TYPE, INT_TYPE, FLOAT_TYPE, DOUBLE_TYPE };

int npd = DefNumPD;	// Number of Points per Dimention
int nk = DefNumI;	// Number of Iterations
int expi = DefExp;	// Export Data

struct timeval startwtime, endwtime;

void help(int argc, char *argv[]);
void export_data(char *G);
void printMatrix(void *m, int r, int c, int elem_type, char *name);


int main(int argc, char* argv[])
{
	help(argc, argv);
	printf("Running with values n=%i, k=%i, o=%i\n", npd, nk, expi);

	srand((unsigned int)time(NULL));

	// Generate random point set
	printf("Generating random data set. ");
	gettimeofday(&startwtime, NULL);
	int *G = (int *)malloc(npd * npd * sizeof(int));
	for (int i = 0; i < npd * npd; i++) {
		if (rand() < (RAND_MAX) / 2)
			*(G + i) = -1;
		else
			*(G + i) = 1;
	}
	gettimeofday(&endwtime, NULL);
	double p_time = (double)((endwtime.tv_usec - startwtime.tv_usec) / 1.0e6 + endwtime.tv_sec - startwtime.tv_sec);
	printf("DONE in %fsec!\n", p_time);

	double weight_matrix[5][5] = { {0.004, 0.016, 0.026, 0.016, 0.004},
									{0.016, 0.071, 0.117, 0.071, 0.016},
									{0.026, 0.117, 0.000, 0.117, 0.026},
									{0.016, 0.071, 0.117, 0.071, 0.016},
									{0.004, 0.016, 0.026, 0.016, 0.004} };

	// Run Ising model evolution
	printf("Running Ising Model Evolution. ");
	gettimeofday(&startwtime, NULL);
	if (!expi) {
		ising(G, &weight_matrix[0][0], nk, npd);
		gettimeofday(&endwtime, NULL);
		p_time = (double)((endwtime.tv_usec - startwtime.tv_usec) / 1.0e6 + endwtime.tv_sec - startwtime.tv_sec);
		printf("DONE in %fsec!\n", p_time);
	}
	else {	// export data
		char *G_out = (char*)malloc(npd * npd * nk * sizeof(char));
		printf("Saving data of each iteration. This will take some time. ");
		for (int i = 0; i < nk; i++) {	// save data of each iteration to export them for animation
			ising(G, &weight_matrix[0][0], 1, npd);
			for (int j = 0; j < npd*npd; j++)	// copy data to export them later
				*(G_out + i * npd*npd + j) = (char)*(G + j);
		}
		gettimeofday(&endwtime, NULL);
		p_time = (double)((endwtime.tv_usec - startwtime.tv_usec) / 1.0e6 + endwtime.tv_sec - startwtime.tv_sec);
		printf("DONE in %fsec!\n", p_time);
		//  Export data to output.bin
		printf("Exporting data to output.bin. ");
		gettimeofday(&startwtime, NULL);
		export_data(G_out);
		p_time = (double)((endwtime.tv_usec - startwtime.tv_usec) / 1.0e6 + endwtime.tv_sec - startwtime.tv_sec);
		printf("DONE in %fsec!\n", p_time);
		free(G_out);
	}

	printf("Exiting\n");
	free(G);
	return 0;
}

void help(int argc, char *argv[])
{
	if (argc > 1) {
		for (int i = 1; i < argc; i += 2) {
			if (*argv[i] == '-') {
				if (*(argv[i] + 1) == 'n')
					npd = atoi(argv[i + 1]);
				else if (*(argv[i] + 1) == 'k')
					nk = atoi(argv[i + 1]);
				else if (*(argv[i] + 1) == 'o') {
					expi = atoi(argv[i + 1]);
				}
				else {
					help(1, argv);
					return;
				}
			}
			else {
				help(1, argv);
				return;
			}
		}
		return;
	}
	printf("Flags to use:\n");
	printf("-n [Number]\t:Number of points per dimention (default:%i)\n", DefNumPD);
	printf("-k [Iterations]\t:Number of iterations (default: %i)\n", DefNumI);
	printf("-o [0|1]\t:Export each iteration to output.bin (default: %i)\n", DefExp);
}

void export_data(char *G)
{
	FILE *f = fopen("output.bin", "wb");
	fwrite(&npd, sizeof(int), 1, f);
	fwrite(&nk, sizeof(int), 1, f);
	fwrite(G, sizeof(char), npd*npd*nk, f);
	fclose(f);
}

void printMatrix(void *m, int r, int c, int elem_type, char *name)
{
	printf("Matrix %s:\n", name);
	for (int i = 0; i < r; i++) {
		printf("{ ");
		for (int j = 0; j < c; j++) {
			if (elem_type == CHAR_TYPE)
				printf("%c, ", *((char*)m + i * c + j));
			else if (elem_type == INT_TYPE)
				printf("%i, ", *((int*)m + i * c + j));
			else if (elem_type == FLOAT_TYPE)
				printf("%f, ", *((float*)m + i * c + j));
			else if (elem_type == DOUBLE_TYPE)
				printf("%lf, ", *((double*)m + i * c + j));
		}
		printf("}\n");
	}
}