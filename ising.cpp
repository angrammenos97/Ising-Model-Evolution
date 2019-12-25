#include "ising.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define WeightMatDim 5	// Weight Matrix Dimension
#define FloatError 1e-6

void update_state(int *G, double influence)
{
	if (influence > FloatError)			// apply threshold for floating point error
		*G = 1;
	else if (influence < -FloatError)	// apply threshold for floating point error
		*G = -1;
	else
		return;
}

int same_matrix(void* A, void* B, int elemSize, int numElem)
{
	for (int i = 0; i < elemSize*numElem; i++)
		if (*((char*)A + i) != *((char*)B + i))
			return 0;
	return 1;
}

void ising(int *G, double *w, int k, int n)
{
	// Initialize memory
	int *G_next = (int*)malloc(n*n * sizeof(int));	// second array to swap pointers
	memcpy(G_next, G, n*n * sizeof(int));

	int i = 0;
	for (i = 0; i < k; i++) {	// for every k iteration
		for (int r_next = 0; r_next < n; r_next++) {	// for every row of the n*n space			
			for (int c_next = 0; c_next < n; c_next++) {	// for every point of the row of the n*n space
				double influence = 0.0;			// weighted influence of neighbors
				for (int x = -2; x <= 2; x++) {	// for every row of weight matrix
					int r = (r_next + x + n) % n;	// wrap around top with bottom
					for (int y = -2; y <= 2; y++) {	// for every weight of a row in weight matrix
						int c = (c_next + y + n) % n;	// wrap around left with right
						influence += *(G + r * n + c) * *(w + (x + 2) * WeightMatDim + (y + 2));	// +2 cause of the x and y offet
					}// for x < WeightMatDim
				}// for y < WeightMatDim
				update_state((G_next + r_next * n + c_next), influence);
			}// for c_next < n
		}// for r_next < n
		// Swap pointers
		int *ptri = G;
		G = G_next;
		G_next = ptri;
		// Exit if nothing changed
		if (same_matrix(G, G_next, sizeof(int), n*n))	
			break;
	}// for i < k

	if ((i % 2) == 1)
		memcpy(G_next, G, n*n * sizeof(int));

	// Free memory
	//free(G_next);
	//o anastasis gamietai piklas
}