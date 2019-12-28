#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "sys/time.h"

#define DefNumPD 517	// Default Number of Points per Dimension
#define DefNumI 10		// Default Number of Iterations
#define DefExp 0		// Default Value to Export Data (0 = False)
#define DefLastF 0		// Default Export Only Last Frame (0 = False)

enum Data_Types { CHAR_TYPE, INT_TYPE, FLOAT_TYPE, DOUBLE_TYPE };

char *input_file = NULL;
int npd = DefNumPD;	// Number of Points per Dimension
int nk = DefNumI;	// Number of Iterations
int expi = DefExp;	// Export Data
int last_frame = DefLastF;

struct timeval startwtime, endwtime;

void ising(int* G, double* w, int k, int n);
void help(int argc, char *argv[]);
void export_data(int *G, int elemNum);
void import_data(int *G);
void printMatrix(void *m, int r, int c, int elem_type, char *name);


int main(int argc, char* argv[])
{
	help(argc, argv);
	printf("Running with values n=%i, k=%i, o=%i, l=%i\n", npd, nk, expi, last_frame);

	srand((unsigned int)time(NULL));
	double p_time;

	// Generate random/Import point set
	int *G = (int *)malloc(npd * npd * sizeof(int));
	if (input_file == NULL) {
		printf("Generating random data set. ");
		gettimeofday(&startwtime, NULL);

		for (int i = 0; i < npd * npd; i++) {
			if (rand() < (RAND_MAX) / 2)
				*(G + i) = -1;
			else
				*(G + i) = 1;
		}
		gettimeofday(&endwtime, NULL);
		p_time = (double)((endwtime.tv_usec - startwtime.tv_usec) / 1.0e6 + endwtime.tv_sec - startwtime.tv_sec);
		printf("DONE in %fsec!\n", p_time);
	}
	else {
		gettimeofday(&startwtime, NULL);
		import_data(G);
		p_time = (double)((endwtime.tv_usec - startwtime.tv_usec) / 1.0e6 + endwtime.tv_sec - startwtime.tv_sec);
		printf("DONE in %fsec!\n", p_time);
	}

	double weight_matrix[5][5] = { {0.004, 0.016, 0.026, 0.016, 0.004},
									{0.016, 0.071, 0.117, 0.071, 0.016},
									{0.026, 0.117, 0.000, 0.117, 0.026},
									{0.016, 0.071, 0.117, 0.071, 0.016},
									{0.004, 0.016, 0.026, 0.016, 0.004} };

	// Run Ising model evolution
	printf("Running Ising Model Evolution. ");
	gettimeofday(&startwtime, NULL);
	if (!expi && !last_frame) {
		ising(G, &weight_matrix[0][0], nk, npd);
		gettimeofday(&endwtime, NULL);
		p_time = (double)((endwtime.tv_usec - startwtime.tv_usec) / 1.0e6 + endwtime.tv_sec - startwtime.tv_sec);
		printf("DONE in %fsec!\n", p_time);
	}
	else if (expi) {	// export data
		printf("Saving data of each iteration. ");
		int *G_out = (int*)malloc(npd * npd * (nk + 1) * sizeof(int));
		memcpy(G_out, G, npd*npd * sizeof(int));	// copy data to export them later		
		for (int i = 1; i < (nk + 1); i++) {	// save data of each iteration to export them for animation
			ising(G, &weight_matrix[0][0], 1, npd);
			memcpy((G_out + i * npd*npd), G, npd*npd * sizeof(int));	// copy data to export them later			
		}
		//  Export data to output.bin
		export_data(G_out, npd*npd*(nk + 1));
		p_time = (double)((endwtime.tv_usec - startwtime.tv_usec) / 1.0e6 + endwtime.tv_sec - startwtime.tv_sec);
		printf("DONE in %fsec!\n", p_time);
		free(G_out);
	}
	else {
		ising(G, &weight_matrix[0][0], nk, npd);
		printf("Saving last iteration. ");
		export_data(G, npd*npd);
		p_time = (double)((endwtime.tv_usec - startwtime.tv_usec) / 1.0e6 + endwtime.tv_sec - startwtime.tv_sec);
		printf("DONE in %fsec!\n", p_time);
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
				if (*(argv[i] + 1) == 'f')
					input_file = argv[i + 1];
				else if (*(argv[i] + 1) == 'n')
					npd = atoi(argv[i + 1]);
				else if (*(argv[i] + 1) == 'k')
					nk = atoi(argv[i + 1]);
				else if (*(argv[i] + 1) == 'o')
					expi = atoi(argv[i + 1]);
				else if (*(argv[i] + 1) == 'l') {
					last_frame = atoi(argv[i + 1]);
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
	printf("-f [File]\t:Input file of points\n");
	printf("-n [Number]\t:Number of points per dimension (default:%i)\n", DefNumPD);
	printf("-k [Iterations]\t:Number of iterations (default: %i)\n", DefNumI);
	printf("-o [0|1]\t:Export each iteration to output*.bin (default: %i)\n", DefExp);
	printf("-l [0|1]\t:Export last iteration to output*.bin (default: %i)\n", DefLastF);

}

void export_data(int *G, int totalSize)
{
	int tmp_k = nk + 1;
	char *out_file_name = (char*)calloc(100, sizeof(char));
	sprintf(out_file_name, "output-%i-%i.bin", npd, tmp_k);
	printf("Exporting data to %s. ", out_file_name);
	FILE *f = fopen(out_file_name, "wb");
	fwrite(G, sizeof(int), totalSize, f);
	fclose(f);
	free(out_file_name);
}

void import_data(int *G)
{
	printf("Importing data from %s. ", input_file);
	FILE *f = fopen(input_file, "rb");
	fread(G, sizeof(int), npd*npd, f);
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
