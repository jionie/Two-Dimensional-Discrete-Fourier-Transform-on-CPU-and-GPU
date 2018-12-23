#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <fstream>
#include <chrono>
#include <ctime>
#include <cmath>

#include "mpi.h"
#include "complex.h"
#include "input_image.h"


using namespace std;



void separate(Complex *a, int n) {
   Complex* b = new Complex[n / 2];  // get temp heap storage
   for (int i = 0; i < n / 2; i++)    // copy all odd elements to heap storage
      b[i] = a[i * 2 + 1];
   for (int i = 0; i < n / 2; i++)    // copy all even elements to lower-half of a[]
      a[i] = a[i * 2];
   for (int i = 0; i < n / 2; i++)    // copy all odd (from heap) to upper-half of a[]
      a[i + n / 2] = b[i];
   delete[] b;                 // delete heap storage
}

// N must be a power-of-2, or bad things will happen.
void fft2(Complex *X, int N) {
   if (N < 2) {
      // bottom of recursion.
      // Do nothing here, because already X[0] = x[0]
      return;
   }
   else {
      separate(X, N);      // all evens to lower half, all odds to upper half
      fft2(X, N / 2);   // recurse even items
      fft2(X + N / 2, N / 2);   // recurse odd  items
      
      // combine results of two half recursions
      for (int k = 0; k < N / 2; k++) {
         Complex e = X[k];   // even
         Complex o = X[k + N / 2];   // odd
         
         // w is the "twiddle-factor"
         Complex w((float)cos(2 * M_PI * k / N), (float)-sin(2 * M_PI * k / N));
         X[k] = e + w * o;
         X[k + N / 2] = e - w * o;
      }
   }
}

void idft2(Complex *X, int N) {
   // this N should be width
   Complex *tmp = (Complex *)malloc(N * sizeof(Complex));
   for (int n = 0; n < N; ++n) {
      tmp[n].real=0;
      tmp[n].imag=0;
      for (int k = 0; k < N; ++k) {
         Complex w((float)cos(2 * M_PI * k * n / N), -(float)sin(2 * M_PI * k * n / N));
         tmp[n] = tmp[n] + X[k] * w;
      }
      tmp[n].real /= N;
      tmp[n].imag /= N;
   }
   for (int p=0; p<N; ++p){
      X[p] = tmp[p];
   }
   free(tmp);
}


void reverseMatrix(Complex *X, int width, int height) {
   Complex *tmp = (Complex *)malloc(width*height * sizeof(Complex));
   for (int i=0; i<width; i++){
      for (int j=0; j<height; j++){
         tmp[j*width+i] = X[i*width+j];
      }
   }
   for (int i=0; i<width; i++){
      for (int j=0; j<height; j++){
         X[j*width+i] = tmp[j*width+i];
      }
   }
   free (tmp);
}


int main(int argc, char* argv[]) {
   // For total n ranks, let NO.i rank do (N/n)*i - (N/n)*(i+1) row. 
   // Then communicate to get the values of (N/n)*i - (N/n)*(i+1) col.
   // Next, let NO.i rank do (N/n)*i - (N/n)*(i+1) col calculation.
   int rank, size;  
   int dir = (strcmp(argv[1], "forward") == 0) ? 1 : -1;
   char *inputFile = argv[2];
   char *outputFile = argv[3];
   InputImage inImage(inputFile);
   int width = inImage.get_width();
   int height = inImage.get_height();
   //cout << "width = " << width << ", height = " << height << '\n';
   Complex *h = inImage.get_image_data();
   // h is a array with length width*height
   // we need to access the h[i][j] by h[i*width+j], in which i is height, j is width

   // --------------------------- MPI Start --------------------------------
   MPI_Init(&argc, &argv);
   MPI_Comm_rank(MPI_COMM_WORLD, &rank);
   MPI_Comm_size(MPI_COMM_WORLD, &size);
   MPI_Request request, request1,request2;

   //cout << "This is rank "<<rank<<". Total size is "<<size<<endl;

   int divided = height/size;
   int remain = height%size;
   int line_per_rank;
   int idx_begin;
   int total_cur_rank;
   int recv_per_rank;
   // assign line_per_rank, idx_begin based on rank
   if (rank>=remain){
      line_per_rank = divided;
      idx_begin = remain*(divided+1)+(rank-remain)*divided;
   }else{
      line_per_rank = divided+1;
      idx_begin = rank*(divided+1);
   }

   chrono::steady_clock::time_point tStart;
   if (rank==0){
      //cout << "Start clokcing...\n";
      tStart = chrono::steady_clock::now();
   }

   if (dir == 1) {
   // dir=1 means forward
      //cout << "Enter Forward"<<"This is rank "<<rank<<endl;
      // ------------first round calculation of row----------------
      for (int ht = idx_begin; ht<idx_begin+line_per_rank; ht++){
         // h[ht*width] --- h[ht*width+width-1]
         // N = width
         //cout<<"FIRST ROUND fft  No. "<<ht<<endl;
         fft2(h+ht*width, width);
      }
      
      // ------------ intermid result sending and receive ---------
      if (rank>=remain){
         total_cur_rank = divided*width;
      }else{
         total_cur_rank = (divided+1)*width;
      }


      Complex *Sec = (Complex *)malloc(total_cur_rank * sizeof(Complex));
      if (rank!=0){
         // rank not 0, send the result to rank 0, receive data into Sec.

         MPI_Send(&h[idx_begin*width], sizeof(Complex) *total_cur_rank, MPI_CHAR, 0, 0, MPI_COMM_WORLD);

         MPI_Recv(&Sec[0], sizeof(Complex) *total_cur_rank, MPI_CHAR, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

      }
      if (rank==0){
         // intermid result collection
         Complex *MID = (Complex *)malloc(width*height * sizeof(Complex));

         // initial the data for rank 0 itself
         for (int i=0; i<total_cur_rank; i++){
            MID[i] = h[i];
         }

         // receive from other rank
         int point = total_cur_rank;
         for (int k=1; k<size; k++){
            if (k>=remain){
               recv_per_rank = divided*width;
            }else{
               recv_per_rank = (divided+1)*width;
            }
            MPI_Recv(&MID[point], sizeof(Complex) *recv_per_rank, MPI_CHAR, k, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            point += recv_per_rank;
         }

         // reverse the matrix
         reverseMatrix(MID, width, height);

         // ------------------------------------------TEST
         // ofstream myfile(outputFile);
         // if (myfile.is_open()) {
         //    for (int i = 0; i < height; ++i) {
         //       for (int j = 0; j < width - 1; ++j) {
         //          myfile << MID[i * width + j].real << ',';
         //       }
         //       myfile << MID[i * width + width - 1].real << '\n';
         //    }
         // }
         // myfile.close();
         // -----------------------------------------------
         // Send back to each rank the reversed data (col)
         point = total_cur_rank;
         for (int k=1; k<size; k++){
            if (k>=remain){
               recv_per_rank = divided*width;
            }else{
               recv_per_rank = (divided+1)*width;
            }
            //cout<< "SENDING to RANK "<< k <<". TOTAL ELEMENT "<< recv_per_rank<< endl;
            MPI_Send(&MID[point], sizeof(Complex) *recv_per_rank, MPI_CHAR, k, 0, MPI_COMM_WORLD);
            point += recv_per_rank;
         }

         for (int i=0; i<total_cur_rank; i++){
            Sec[i] = MID[i];
         }
         free(MID);
      }

      // Now Sec is a matrix with line_per_rank row. (which is colum before)

      if (rank==1){
         ofstream myfile(outputFile);
         if (myfile.is_open()) {
            for (int i = 0; i < height/size; ++i) {
               for (int j = 0; j < width; ++j) {
                  myfile << Sec[i * width + j].real << ',';
               }
               myfile << Sec[i * width + width - 1].real << '\n';
            }
         }
         myfile.close();
      }

      for (int row = 0; row<line_per_rank; row++){
         // Sec[row*width] --- Sec[row*width+width-1]
         // N = width
         fft2(Sec+row*width, width);
      }


      // Send the final result (final col as present row) to rank 0
      if (rank!=0){
         MPI_Send(&Sec[0], sizeof(Complex) *total_cur_rank, MPI_CHAR, 0, 1, MPI_COMM_WORLD);
      }
      if (rank==0){
         Complex *RES = (Complex *)malloc(width*height * sizeof(Complex));
         for (int i=0; i<total_cur_rank; i++){
            RES[i] = Sec[i];
         }
         int point2 = total_cur_rank;
         for (int k=1; k<size; k++){
            if (rank>=remain){
               recv_per_rank = divided*width;
            }else{
               recv_per_rank = (divided+1)*width;
            }
            MPI_Recv(&RES[point2], sizeof(Complex) *recv_per_rank, MPI_CHAR, k, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            point2 += recv_per_rank;
         }
         // reverse the matrix
         reverseMatrix(RES, width, height);
         chrono::steady_clock::time_point tEnd = chrono::steady_clock::now();
         chrono::duration<double> time_span = chrono::duration_cast<chrono::duration<double> >(tEnd - tStart);
         cout << "Time ellipsed: " << time_span.count() << " seconds... \n";
         //cout << "Printing results...\n";
         // ofstream myfile(outputFile);
         // if (myfile.is_open()) {
         //    for (int i = 0; i < height; ++i) {
         //       for (int j = 0; j < width - 1; ++j) {
         //          myfile << RES[i * width + j].real << ',';
         //       }
         //       myfile << RES[i * width + width - 1].real << '\n';
         //    }
         // }
         // myfile.close();
         inImage.save_image_data(outputFile, RES, width, height);
         free(h);
      }
      free(Sec);
   }


   else{
   // dir=0 means backward
      //cout << "Enter Backward"<<"This is rank "<<rank<<endl;
      // ------------first round calculation of row----------------
      reverseMatrix(h, width, height);
      for (int ht = idx_begin; ht<idx_begin+line_per_rank; ht++){
         // h[ht*width] --- h[ht*width+width-1]
         // N = width
         //cout<<"FIRST ROUND fft  No. "<<ht<<endl;
         idft2(h+ht*width, width);
      }
      
      // ------------ intermid result sending and receive ---------
      if (rank>=remain){
         total_cur_rank = divided*width;
      }else{
         total_cur_rank = (divided+1)*width;
      }


      Complex *Sec = (Complex *)malloc(total_cur_rank * sizeof(Complex));
      if (rank!=0){
         // rank not 0, send the result to rank 0, receive data into Sec.

         MPI_Send(&h[idx_begin*width], sizeof(Complex) *total_cur_rank, MPI_CHAR, 0, 0, MPI_COMM_WORLD);

         MPI_Recv(&Sec[0], sizeof(Complex) *total_cur_rank, MPI_CHAR, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      }
      if (rank==0){
         // intermid result collection
         Complex *MID = (Complex *)malloc(width*height * sizeof(Complex));

         // initial the data for rank 0 itself
         for (int i=0; i<total_cur_rank; i++){
            MID[i] = h[i];
         }

         // receive from other rank
         int point = total_cur_rank;
         for (int k=1; k<size; k++){
            if (k>=remain){
               recv_per_rank = divided*width;
            }else{
               recv_per_rank = (divided+1)*width;
            }
            MPI_Recv(&MID[point], sizeof(Complex) *recv_per_rank, MPI_CHAR, k, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            point += recv_per_rank;
         }

         // ------------------------------------------TEST
         // ofstream myfile(outputFile);
         // if (myfile.is_open()) {
         //    for (int i = 0; i < height; ++i) {
         //       for (int j = 0; j < width - 1; ++j) {
         //          myfile << MID[i * width + j].real << ',';
         //       }
         //       myfile << MID[i * width + width - 1].real << '\n';
         //    }
         // }
         // myfile.close();
         // -----------------------------------------------

         // reverse the matrix
         reverseMatrix(MID, width, height);

         // Send back to each rank the reversed data (col)
         point = total_cur_rank;
         for (int k=1; k<size; k++){
            if (k>=remain){
               recv_per_rank = divided*width;
            }else{
               recv_per_rank = (divided+1)*width;
            }
            MPI_Send(&MID[point], sizeof(Complex) *recv_per_rank, MPI_CHAR, k, 0, MPI_COMM_WORLD);
            point += recv_per_rank;
         }

         for (int i=0; i<total_cur_rank; i++){
            Sec[i] = MID[i];
         }
         free(MID);
      }

      // Now Sec is a matrix with line_per_rank row. (which is colum before)

      for (int row = 0; row<line_per_rank; row++){
         // Sec[row*width] --- Sec[row*width+width-1]
         // N = width
         idft2(Sec+row*width, width);
      }


      // Send the final result (final col as present row) to rank 0
      if (rank!=0){
         MPI_Send(&Sec[0], sizeof(Complex) *total_cur_rank, MPI_CHAR, 0, 1, MPI_COMM_WORLD);
      }
      if (rank==0){
         Complex *RES = (Complex *)malloc(width*height * sizeof(Complex));
         for (int i=0; i<total_cur_rank; i++){
            RES[i] = Sec[i];
         }
         int point2 = total_cur_rank;
         for (int k=1; k<size; k++){
            if (rank>=remain){
               recv_per_rank = divided*width;
            }else{
               recv_per_rank = (divided+1)*width;
            }
            MPI_Recv(&RES[point2], sizeof(Complex) *recv_per_rank, MPI_CHAR, k, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            point2 += recv_per_rank;
         }
         // reverse the matrix
         //reverseMatrix(RES, width, height);
         chrono::steady_clock::time_point tEnd = chrono::steady_clock::now();
         chrono::duration<double> time_span = chrono::duration_cast<chrono::duration<double> >(tEnd - tStart);
         cout << "Time ellipsed: " << time_span.count() << " seconds... \n";
         //cout << "Printing results...\n";
         // ofstream myfile(outputFile);
         // if (myfile.is_open()) {
         //    for (int i = 0; i < height; ++i) {
         //       for (int j = 0; j < width - 1; ++j) {
         //          myfile << RES[i * width + j].real << ',';
         //       }
         //       myfile << RES[i * width + width - 1].real << '\n';
         //    }
         // }
         // myfile.close();
         inImage.save_image_data_real(outputFile, RES, width, height);
         free(h);
      }
      free(Sec);

   }

   MPI_Finalize();

   return 0;
} 
