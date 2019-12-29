//////////////// Ising Code Here ////////////////
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#define WeightMatDim 5	// Weight Matrix Dimension
#define FloatError 1e-6	// Float error
#define TileSize 32		// Size of tiles partitioning the matrix - each tile calculates TileSize x TileSize moments
#define NumberOfRows 8	// Rows of each block of threads - each block is of size NumberOfRows x TileSize

__device__ int state_d; // device parameter to hold if iterations should proceed
int state;				// corresponding host parameter

__global__ void same_matrix(void* A, void* B, int elemSize, int numElem)
{
	state_d = 1;
	for (int i = 0; i < elemSize * numElem; i++)
		if (*((char*)A + i) != *((char*)B + i)) {
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
					influence += *(G_d + r * n + c) * *(w_d + (i + 2) * WeightMatDim + (t + 2));	// +2 cause of the i and t offset
				}// for t < WeightMatDim
			}// for i < WeightMatDim

			  /*Update state for current point*/
			if (influence > FloatError)			// apply threshold for floating point error
				*(GNext_d + y * n + x) = 1;
			else if (influence < -FloatError)	// apply threshold for floating point error
				*(GNext_d + y * n + x) = -1;
			else								// stay the same
				*(GNext_d + y * n + x) = *(G_d + y * n + x);

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
	dim3 dimBlock(TileSize, NumberOfRows);
	dim3 dimGrid((n + dimBlock.x - 1) / dimBlock.x, (n + dimBlock.y - 1) / dimBlock.y);

	/*--------------------------------------------------------------------------------*/
	for (int i = 0; i < k; ++i) { // For every iteration
		calculateFrame << < dimGrid, dimBlock >> > (G_d, GNext_d, w_d, n);
		same_matrix << < 1, 1 >> > ((void*)G_d, (void*)GNext_d, sizeof(int), n * n);

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