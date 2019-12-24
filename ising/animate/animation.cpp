#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <core.hpp>
#include <highgui.hpp>

#define mSPF  1000	// milliseconds per frame

using namespace cv;

void create_output();
void help(int argc, char *argv[]);

FILE *f = NULL;

int main(int argc, char *argv[])
{
	help(argc, argv);

	int *bufi = (int*)malloc(sizeof(int));
	fread(bufi, sizeof(int), 1, f);
	int n = *bufi;
	fread(bufi, sizeof(int), 1, f);
	int k = *bufi;
	free(bufi);
	char *all_frames = (char*)malloc(n * n * k * sizeof(char));
	fread(all_frames, sizeof(char), n*n*k, f);
	fclose(f);

	printf("n=%i k=%i\n", n, k);
	Mat frame(n, n, CV_8UC1);
	int millis_per_frame = mSPF;
	for (int i = 0; i < k; i++) {
		memcpy(frame.ptr(0), (all_frames + i * n*n), n*n);
		printf("frame #%i\n", i);
		imshow("Frame", frame);
		int c = waitKey(millis_per_frame);
		if (c == 32) { // space button
			if (millis_per_frame == mSPF)	// pause
				millis_per_frame = -1;
			else
				millis_per_frame = mSPF;	// play

		}
		else if (c == 112) { // p button
			millis_per_frame = -1;			// pause
			i -= 2;
			if (i < 0)
				i = -1;						// go back a frame
		}
		else if (c == 113 || c == 27) // q or esc button
			break;							// exit
	}


	printf("Exiting...\n");
	destroyAllWindows();
	frame.release();
	free(all_frames);
	return 0;
}

void help(int argc, char *argv[])
{
	if (argc > 1) {
		if (*argv[1] == '-' && *(argv[1] + 1) == 'r') {
			create_output();
			exit(0);
		}
		else
			f = fopen(argv[1], "rb");
	}
	else {
		printf("No input file! You can create a random one with flag -r.\n");
		exit(-1);
	}
	printf("Key mapping:\n");
	printf("space\t: Play/Pause\n");
	printf("p\t: Previous frame\n");
	printf("esc|q\t: Exit\n");
	printf("Other\t: Goes to next frame\n");
	printf("Note: When press p the playback pauses.\n");
}

void create_output()
{
	printf("Creating random frames in random.bin.\n");
	srand((unsigned int)time(NULL));
	FILE *f = fopen("random.bin", "wb");
	int n = 512;
	int k = 20;
	char minus_one = -1;
	char plus_one = 1;
	fwrite(&n, sizeof(int), 1, f);
	fwrite(&k, sizeof(int), 1, f);
	for (int i = 0; i < n*n*k; i++) {
		if (rand() < RAND_MAX / 2)
			fwrite(&minus_one, sizeof(char), 1, f);
		else
			fwrite(&plus_one, sizeof(char), 1, f);
	}
}
