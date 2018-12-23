#include <iostream> 
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <thread>
#include <fstream>
#include <string>
#include <chrono>
#include <ctime>
#include <cmath>
#include <cstring>

#include "complex.h"
#include "input_image.h"

#define M_PI 3.14159265358979323846 // Pi constant with double precision
#define NUM_THREADS 32

using namespace std;

void dft_thread_row(Complex *x, Complex *X, Complex *omega, int N, int head, int tail);
void dft_thread_row2(Complex *x, Complex *X, Complex *omega, int N, int head, int tail);
void idft_thread_row(Complex *X, Complex *x, int N, int head, int tail);
void idft_thread_col(Complex *X, Complex *x, int N, int width, int head, int tail);

void dft_thread_row(Complex *x, Complex *X, Complex *omega, int N, int head, int tail) {
	while (head < tail) {
		for (int i = 0; i < N; ++i) {
			X[head * N] = X[head * N] + x[head * N + i];
		}
		if (N % 2 == 0) { // even number of elements
			for (int k = 1; k < N / 2; ++k) {
				for (int n = 0; n < N; ++n) {
					//Complex w((float)cos(2 * M_PI * k * n / N), (float)-sin(2 * M_PI * k * n / N));
					X[head * N + k] = X[head * N + k] + x[head * N + n] * omega[n * N + k];
					X[(head + 1) * N - k] = X[head * N + k].conj();
				}
			}
			for (int n = 0; n < N; ++n) {
				Complex w((float)cos(M_PI * n));
				X[head * N + N / 2] = X[head * N + N / 2] + x[head * N + n] * w;
			}
		}
		else {
			for (int k = 1; k < N / 2; ++k) {
				for (int n = 0; n < N; ++n) {
					//Complex w((float)cos(2 * M_PI * k * n / N), (float)-sin(2 * M_PI * k * n / N));
					X[head * N + k] = X[head * N + k] + x[head * N + n] * omega[n * N + k];
					X[(head + 1) * N - k] = X[head * N + k].conj();
				}
			}
		}
		head++;
	}
}

void dft_thread_row2(Complex *x, Complex *X, Complex *omega, int N, int head, int tail) {
	while (head < tail) {
		for (int k = 0; k < N; ++k) {
			for (int n = 0; n < N; ++n) {
				//Complex w((float)cos(2 * M_PI * k * n / N), (float)-sin(2 * M_PI * k * n / N));
				X[head * N + k] = X[head * N + k] + x[head * N + n] * omega[n * N + k];
			}
		}
		head++;
	}
}

void idft_thread_row(Complex *X, Complex *x, int N, int head, int tail) {
	while (head < tail) {
		for (int n = 0; n < N; ++n) {
			for (int k = 0; k < N; ++k) {
				Complex w((float)cos(2 * M_PI * k * n / N), (float)sin(2 * M_PI * k * n / N));
				x[head * N + n] = x[head * N + n] + X[head * N + k] * w;
			}
			x[head * N + n].real /= N;
			x[head * N + n].imag /= N;
		}
		head++;
	}
}

void idft_thread_col(Complex *X, Complex *x, int N, int width, int head, int tail) {
	while (head < tail) {
		for (int n = 0; n < N; ++n) {
			for (int k = 0; k < N; ++k) {
				Complex w((float)cos(2 * M_PI * k * n / N), (float)sin(2 * M_PI * k * n / N));
				x[n * width + head] = x[n * width + head] + X[k * width + head] * w;
			}
			x[n * width + head].real /= N;
			x[n * width + head].imag /= N;
		}
		head++;
	}
}

void trans(Complex* h, int width, int height)
{
	for (int i = 0; i < height; ++i) {
		for (int j = 0; j < width; ++j) {
			if (j > i) {
				Complex temp = h[i * width + j];
				h[i * width + j] = h[j * width + i];
				h[j * width + i] = temp;
			}
		}
	}	
}

int main(int argc, char* argv[]) {
	int dir = (strcmp(argv[1], "forward") == 0) ? 1 : -1;
	char *inputFile = argv[2];
	char *outputFile = argv[3];
	InputImage inImage(inputFile);
	int width = inImage.get_width();
	int height = inImage.get_height();
	cout << "width = " << width << ", height = " << height << '\n';

	if (dir == 1) { // forward
		Complex *h = inImage.get_image_data();

		cout << "Start clocking...\n";
		chrono::steady_clock::time_point tStart = chrono::steady_clock::now();
		Complex *X = (Complex *)malloc(width * height * sizeof(Complex));
		Complex *H = (Complex *)malloc(width * height * sizeof(Complex));
		Complex *omega = (Complex *)malloc(width * height * sizeof(Complex));
		for (int i = 0; i < width * height; ++i) {
			X[i] = 0;
			H[i] = 0;
		}
		for (int k = 0; k < width; ++k) {
			for (int n = 0; n < width; ++n) {
				omega[k * width + n] = Complex((float)cos(2 * M_PI * k * n / width), (float)-sin(2 * M_PI * k * n / width));
			}
		}
		
		//vector<thread> thread_pool;
		thread thread_pool[NUM_THREADS];
		//thread_pool.reserve(NUM_THREADS);
		//rows
		int row_per_thread = height / NUM_THREADS;
		for (int i = 0; i < NUM_THREADS; ++i) {
			int head = i * row_per_thread;
			int tail = (i == NUM_THREADS - 1) ? height : (i + 1) * row_per_thread;
			thread_pool[i] = thread(dft_thread_row, h, X, omega, width, head, tail);
		}
		for (int i = 0; i < NUM_THREADS; ++i)
			thread_pool[i].join();

		trans(X, width, height);

		//columns
		for (int i = 0; i < NUM_THREADS; ++i) {
			int head = i * row_per_thread;
			int tail = (i == NUM_THREADS - 1) ? width : (i + 1) * row_per_thread;
			thread_pool[i] = thread(dft_thread_row2, X, H, omega, height, head, tail);
		}
		for (int i = 0; i < NUM_THREADS; ++i)
			thread_pool[i].join();

		trans(H, height, width);

		chrono::steady_clock::time_point tEnd = chrono::steady_clock::now();
		chrono::duration<double> time_span = chrono::duration_cast<chrono::duration<double>>(tEnd - tStart);
		cout << "Time ellipsed: " << time_span.count() << " seconds... \n";
		cout << "Printing results...\n";
		
		inImage.save_image_data(outputFile, H, width, height);

		free(X); free(H);
	}
	else { // reverse DFT
		Complex *H = inImage.get_image_data();
		Complex *X = (Complex *)malloc(width * height * sizeof(Complex));
		Complex *h = (Complex *)malloc(width * height * sizeof(Complex));
		for (int i = 0; i < width; ++i) {
			X[i] = 0;
			h[i] = 0;
		}
		cout << "Start clocking...\n";
		chrono::steady_clock::time_point tStart = chrono::steady_clock::now();

		vector<thread> thread_pool;
		thread_pool.reserve(NUM_THREADS);
		//columns
		int col_per_thread = width / NUM_THREADS;
		for (int i = 0; i < width * height; ++i) {
			h[i] = 0;
			X[i] = 0;
		}
		for (int i = 0; i < NUM_THREADS; ++i) {
			int head = i * col_per_thread;
			int tail = (i == NUM_THREADS - 1) ? width : (i + 1) * col_per_thread;
			thread_pool.push_back(thread(idft_thread_col, H, X, height, width, head, tail));
		}
		for (int i = 0; i < NUM_THREADS; ++i)
			thread_pool[i].join();
		thread_pool.clear();
		//rows
		int row_per_thread = height / NUM_THREADS;
		for (int i = 0; i < NUM_THREADS; ++i) {
			int head = i * row_per_thread;
			int tail = (i == NUM_THREADS - 1) ? height : (i + 1) * row_per_thread;
			thread_pool.push_back(thread(idft_thread_row, X, h, width, head, tail));
		}
		for (int i = 0; i < NUM_THREADS; ++i)
			thread_pool[i].join();
		thread_pool.clear();

		chrono::steady_clock::time_point tEnd = chrono::steady_clock::now();
		chrono::duration<double> time_span = chrono::duration_cast<chrono::duration<double>>(tEnd - tStart);
		cout << "Time ellipsed: " << time_span.count() << " seconds... \n";
		cout << "Printing results...\n";
		inImage.save_image_data_real(outputFile, h, width, height);
		free(X); free(H);
	}
	cout << "finished...\n";
	return 0;
}