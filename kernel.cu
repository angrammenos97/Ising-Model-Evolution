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

	double weight_matrix[5][5] = {	{0.004, 0.016, 0.026, 0.016, 0.004},
									{0.016, 0.071, 0.117, 0.071, 0.016},
									{0.026, 0.117, 0.000, 0.117, 0.026},
									{0.016, 0.071, 0.117, 0.071, 0.016},
									{0.004, 0.016, 0.026, 0.016, 0.004}	};

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
		printf("Saving data of each iteration. This will take some time. ");
		int *G_out = (int*)malloc(npd * npd * (nk + 1) * sizeof(int));
		memcpy(G_out, G, npd*npd*sizeof(int));	// copy data to export them later		
		for (int i = 1; i < (nk+1); i++) {	// save data of each iteration to export them for animation
			ising(G, &weight_matrix[0][0], 1, npd);
			memcpy((G_out + i*npd*npd), G, npd*npd*sizeof(int));	// copy data to export them later			
		}
		//  Export data to output.bin
		export_data(G_out, npd*npd*(nk+1));
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
	//free(G);
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

//////////////// Ising Code Here ////////////////
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#define WeightMatDim 5	// Weight Matrix Dimension
#define FloatError 1e-6
#define TileSize 32 // Size of tiles partitioning the matrix - each tile calculates TileSize x TileSize moments
#define NumberOfRows 32 // Rows of each block of threads - each block is of size NumberOfRows x TileSize

__device__ int state_d; // Device parameter to hold if iterations should proceed
int state; // corresponding host parameter

__global__ void same_matrix(void* A, void* B, int elemSize, int numElem)
{
	state_d = 1;

	for (int i = 0; i < elemSize * numElem; i++)

		if (*((int*)A + i) != *((int*)B + i)) {
			state_d = 0;
			break;
		}
}

__global__ void calculateFrame(int* G_d, int* GNext_d, double* w_d, int n)
{
	/* Initialize thread coordinates*/
	int x = blockIdx.x * TileSize + threadIdx.x;
	int y = blockIdx.y * TileSize + threadIdx.y;

	/*------------------------------------------*/

	if (x < n) { // Check whether thread x coordinate is out of bounds
		for (int j = 0; j < TileSize && y < n; j += NumberOfRows) { // for every block chunk of tile in the y axis 

			double influence = 0.0;			// weighted influence of neighbors

			for (int i = -2; i <= 2; i++) {	// for every row of weight matrix
				int r = (y + i + n) % n;	// wrap around top with bottom
				for (int t = -2; t <= 2; t++) {	// for every weight of a row in weight matrix
					int c = (x + t + n) % n;	// wrap around left with right
					influence += *(G_d + r * n + c) * *(w_d + (i + 2) * WeightMatDim + (t + 2));	// +2 cause of the x and y offet
				}// for t < WeightMatDim
			}// for i < WeightMatDim


			 /*Update state for current point*/

			if (influence > FloatError)			// apply threshold for floating point error
				GNext_d[y * n + x] = 1;
			else                             	// apply threshold for floating point error
				GNext_d[y * n + x] = -1;

			y += NumberOfRows; // Update y coordinate as we move down the tile

		}// for every j < TileSize && y within G matrix's bounds
	}
}

void ising(int* G, double* w, int k, int n)
{
	/*Declare and initialize memory to use on device*/

	int* G_d, *GNext_d; double* w_d; // Pointers to use for matrix store on device

	cudaMalloc((void**)&G_d, n * n * sizeof(int));
	cudaMalloc((void**)&GNext_d, n * n * sizeof(int));
	cudaMalloc((void**)&w_d, WeightMatDim * WeightMatDim * sizeof(double));

	cudaMemcpy(G_d, G, n * n * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(GNext_d, G, n * n * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(w_d, w, WeightMatDim * WeightMatDim * sizeof(double), cudaMemcpyHostToDevice);

	/*Declare grid and block sizes and compensate for matrix not divided with block size*/

	dim3 dimBlock(NumberOfRows, TileSize);
	dim3 dimGrid((n + dimBlock.x - 1) / dimBlock.x, (n + dimBlock.y - 1) / dimBlock.y);

	/*--------------------------------------------------------------------------------*/

	for (int i = 0; i < k; ++i) { // For every iteration

		calculateFrame <<< dimGrid, dimBlock >>> (G_d, GNext_d, w_d, n);
		same_matrix <<< 1, 1 >>> ((void*)G_d, (void*)GNext_d, sizeof(int), n * n);

		cudaMemcpy(&state, &state_d, sizeof(int), cudaMemcpyDeviceToHost); // Kernel to get flag indicating whether matrices are the same

		// Swap pointers
		int* ptri = G_d;
		G_d = GNext_d;
		GNext_d = ptri;

		// Exit if nothing changed
		if (state) {
			cudaFree(GNext_d);
			cudaFree(w_d);
			break;
		}
	}  // for every i < k

	// copy data from device and cleanup
	cudaMemcpy(G, G_d, n * n * sizeof(int), cudaMemcpyDeviceToHost);
	cudaFree(G_d);
}
/////////////////////////////////////////////////