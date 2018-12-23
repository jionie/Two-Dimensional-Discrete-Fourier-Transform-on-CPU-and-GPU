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
#include <cuda.h>
#include <sstream>



#define M_PI 3.14159265358979323846 // Pi constant with double precision

using namespace std;

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
const float PI = 3.14159265358979323846;
//////////////////////////////////////////////////////////////////////////////////////////////
class Complex_cuda{
public:
    __device__ __host__ Complex_cuda();
    __device__ __host__ Complex_cuda(float r, float i);
    __device__ __host__ Complex_cuda(float r);
    __device__ __host__ Complex_cuda operator+(const Complex_cuda& b) const;
    __device__ __host__ Complex_cuda operator-(const Complex_cuda& b) const;
    __device__ __host__ Complex_cuda operator*(const Complex_cuda& b) const;

    __device__ __host__ Complex_cuda mag() const;
    __device__ __host__ Complex_cuda angle() const;
    __device__ __host__ Complex_cuda conj() const;

    float real;
    float imag;
};

//std::ostream& operator<<(std::ostream& os, const Complex_cuda& rhs);

__device__ __host__ Complex_cuda::Complex_cuda() : real(0.0f), imag(0.0f) {}

__device__ __host__ Complex_cuda::Complex_cuda(float r) : real(r), imag(0.0f) {}

__device__ __host__ Complex_cuda::Complex_cuda(float r, float i) : real(r), imag(i) {}

__device__ __host__ Complex_cuda Complex_cuda::operator+(const Complex_cuda &b) const {
	return Complex_cuda(this->real + b.real, this->imag + b.imag);
}

__device__ __host__ Complex_cuda Complex_cuda::operator-(const Complex_cuda &b) const {
	return Complex_cuda(this->real - b.real, this->imag - b.imag);
}

__device__ __host__ Complex_cuda Complex_cuda::operator*(const Complex_cuda &b) const {
	return Complex_cuda(this->real * b.real - this->imag * b.imag, this->real * b.imag + this->imag * b.real);
}

__device__ __host__ Complex_cuda Complex_cuda::mag() const {
	return Complex_cuda(sqrt(pow(this->real, 2.0) + pow(this->imag, 2.0)));
}

__device__ __host__ Complex_cuda Complex_cuda::angle() const {
	if (this->imag > 0) return Complex_cuda(acos(this->real / this->mag().real));
	if (this->imag < 0) return Complex_cuda(-acos(this->real / this->mag().real));
	if (this->real > 0 && this->imag == 0) return Complex_cuda(0);
	return Complex_cuda(PI);
}

__device__ __host__ Complex_cuda Complex_cuda::conj() const {
	return Complex_cuda(this->real, -this->imag);
}

std::ostream& operator<< (std::ostream& os, const Complex_cuda& rhs) {
    Complex_cuda c(rhs);
    if(fabsf(rhs.imag) < 1e-5) c.imag = 0.0f;
    if(fabsf(rhs.real) < 1e-5) c.real = 0.0f;

    if(c.imag == 0) {
        os << c.real;
    }
    else {
        os << "(" << c.real << "," << c.imag << ")";
    }
    return os;
}

/////////////////////////////////////////////////////////////////////////////////////////////
class InputImage {
public:

    InputImage(const char* filename);
    int get_width() const;
    int get_height() const;

    //returns a pointer to the image data.  Note the return is a 1D
    //array which represents a 2D image.  The data for row 1 is
    //immediately following the data for row 0 in the 1D array
    Complex_cuda* get_image_data() const;

    //use this to save output from forward DFT
    void save_image_data(const char* filename, Complex_cuda* d, int w, int h);
    //use this to save output from reverse DFT
    void save_image_data_real(const char* filename, Complex_cuda* d, int w, int h);

private:
    int w;
    int h;
    Complex_cuda* data;
};


InputImage::InputImage(const char* filename) {
    std::ifstream ifs(filename);
    if(!ifs) {
        std::cout << "Can't open image file " << filename << std::endl;
        exit(1);
    }

    ifs >> w >> h;
    data = new Complex_cuda[w * h];
    for(int r = 0; r < h; ++r) {
        for(int c = 0; c < w; ++c) {
            float real;
            ifs >> real;
            data[r * w + c] = Complex_cuda(real);
        }

    }
}

int InputImage::get_width() const {
    return w;
}

int InputImage::get_height() const {
    return h;
}

Complex_cuda* InputImage::get_image_data() const {
    return data;
}

void InputImage::save_image_data(const char *filename, Complex_cuda *d, int w, int h) {
    std::ofstream ofs(filename);
    if(!ofs) {
        std::cout << "Can't create output image " << filename << std::endl;
        return;
    }

    ofs << w << " " << h << std::endl;

    for(int r = 0; r < h; ++r) {
        for(int c = 0; c < w; ++c) {
            ofs << d[r * w + c] << " ";
        }
        ofs << std::endl;
    }
}

void InputImage::save_image_data_real(const char* filename, Complex_cuda* d, int w, int h) {
    std::ofstream ofs(filename);
    if(!ofs) {
        std::cout << "Can't create output image " << filename << std::endl;
        return;
    }

    ofs << w << " " << h << std::endl;

    for (int r = 0; r < h; ++r) {
        for (int c = 0; c < w; ++c) {
            ofs << d[r * w + c].real << " ";
        }
        ofs << std::endl;
    }
}
/////////////////////////////////////////////////////////////////////////////////////////////




inline void gpuAssert(cudaError_t code, char *file, int line, bool abort=true)
{
    if (code != cudaSuccess) 
    {
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) getchar();
    }
}


int Divup(int a, int b)
{

    if (a % b)
    {
        return a / b + 1; /* add in additional block */
    }
    else
    {
        return a / b; /* divides cleanly */
    }
}


__global__ void fft(Complex_cuda *a, int num_col);
__global__ void ifft(Complex_cuda *a, int num_col);
__device__ void transpose(Complex_cuda *Matrix, int num_col);

__global__ void fft(Complex_cuda *a, Complex_cuda *omega, int num_col)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;

    for(int i=0; i<num_col; i++)  
    {
        //printf("0: i=%d, x=%d, num_col=%d\n", i, x, num_col);
        omega[i] = Complex_cuda(cos(2*M_PI/(float)num_col*i), -sin(2*M_PI/(float)num_col*i));
    }

    //printf("1: x=%d\n", x);
    for(int i=0, j=0; i<num_col; ++i)  
    {
        if (i>j)  
        {
            Complex_cuda tmp = a[x*num_col+i];
            a[x*num_col+i] = a[x*num_col+j];
            a[x*num_col+j] = tmp;
        }
		for(int l=num_col>>1; (j^=l)<l; l>>=1);
	}

    //printf("2: x=%d\n", x);
    for(int l=2; l<=num_col; l<<=1)  
    {
        int m = l/2;
        for(Complex_cuda *p=a; p!=a+num_col; p=p+l)  
        {
            for(int i=0; i<m; ++i)  
            {
                Complex_cuda t = omega[num_col/l*i] * p[x*num_col+m+i];
                p[x*num_col+m+i] = p[x*num_col+i] - t ;
                p[x*num_col+i] = p[x*num_col+i] + t ;
            }
        }
    }
    //printf("3: x=%d\n", x);
    __syncthreads();
    transpose(a, num_col); /* get transpose result and do fft again == do fft on another axis*/
    __syncthreads();

    //printf("4: x=%d\n", x);
    for(int i=0, j=0 ; i<num_col ; ++i)  
    {
        if (i>j)  
        {
            Complex_cuda tmp = a[x*num_col+i];
            a[x*num_col+i] = a[x*num_col+j];
            a[x*num_col+j] = tmp;
        }
		for(int l=num_col>>1; (j^=l)<l; l>>=1);
	}
    //printf("5: x=%d\n", x);
    for(int l=2; l<=num_col; l<<=1)  
    {
        int m = l/2;
        for(Complex_cuda *p=a; p!=a+num_col; p=p+l)  
        {
            for(int i=0; i<m; ++i)  
            {
                Complex_cuda t = omega[num_col/l*i] * p[x*num_col+m+i];
                p[x*num_col+m+i] = p[x*num_col+i] - t ;
                p[x*num_col+i] = p[x*num_col+i] + t ;
            }
        }
    }
    //printf("6: x=%d\n", x);
    transpose(a, num_col);
    __syncthreads();
}

__global__ void ifft(Complex_cuda *a, Complex_cuda *omega_inverse, int num_col)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;

    for(int i=0; i<num_col; ++ i)  {
        omega_inverse[i] = Complex_cuda(cos(2*M_PI/num_col*i), -sin(2*M_PI/num_col*i)).conj();
    }

    for(int i=0, j=0; i<num_col ; ++i)  
    {
        if (i>j)  
        {
            Complex_cuda tmp = a[x*num_col+i];
            a[x*num_col+i] = a[x*num_col+j];
            a[x*num_col+j] = tmp;
        }
		for(int l=num_col>>1; (j^=l)<l; l>>=1);
	}

    for(int l=2; l<=num_col; l<<=1)  
    {
        int m = l/2;
        for(Complex_cuda *p=a; p!=a+num_col; p=p+l)  
        {
            for(int i=0; i<m; ++i)  
            {
                Complex_cuda t = omega_inverse[num_col/l*i] * p[x*num_col+m+i];
                p[x*num_col+m+i] = p[x*num_col+i] - t ;
                p[x*num_col+i] = p[x*num_col+i] + t ;
            }
        }
    }

    __syncthreads();
    transpose(a, num_col); /* get transpose result and do ifft again == do ifft on another axis*/
    __syncthreads();

    for(int i=0, j=0; i<num_col ; ++i)  
    {
        if (i>j)  
        {
            Complex_cuda tmp = a[x*num_col+i];
            a[x*num_col+i] = a[x*num_col+j];
            a[x*num_col+j] = tmp;
        }
		for(int l=num_col>>1; (j^=l)<l; l>>=1);
	}

    for(int l=2; l<=num_col; l<<=1)  
    {
        int m = l/2;
        for(Complex_cuda *p=a; p!=a+num_col; p=p+l)  
        {
            for(int i=0; i<m; ++i)  
            {
                Complex_cuda t = omega_inverse[num_col/l*i] * p[x*num_col+m+i];
                p[x*num_col+m+i] = p[x*num_col+i] - t ;
                p[x*num_col+i] = p[x*num_col+i] + t ;
            }
        }
    }
    transpose(a, num_col);
    __syncthreads();

}

__device__ void transpose(Complex_cuda *Matrix, int num_col)
{    
    const int x = blockIdx.x * blockDim.x + threadIdx.x ;  
    for(int i=x+1; i<num_col; i++)
    {
        Complex_cuda tmp = Matrix[x*num_col+i];        
        Matrix[x*num_col+i]=Matrix[i*num_col+x];        
        Matrix[i*num_col+x]=tmp;
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
        Complex_cuda *h = inImage.get_image_data();
		Complex_cuda *X = (Complex_cuda*)malloc(width * height * sizeof(Complex_cuda));
        Complex_cuda *h_cuda;
        cudaMalloc((void **)&h_cuda, width * height * sizeof(Complex_cuda));
        Complex_cuda* omega = (Complex_cuda *)malloc(width*sizeof(Complex_cuda));
        Complex_cuda* omega_cuda;
        cudaMalloc((void **)&omega_cuda, width * sizeof(Complex_cuda));
		
		for (int i = 0; i < width * height; ++i) {
			X[i] = 0;
		}
		cout << "Start clokcing...\n";
		chrono::steady_clock::time_point tStart = chrono::steady_clock::now();

        int block_size = (height>512)?(512):(height);
        dim3 grid(Divup(height,block_size),1);    
        dim3 block(block_size,1);
        cudaMemcpy(h_cuda, h, width*height*sizeof(Complex_cuda), cudaMemcpyHostToDevice);
        cudaMemcpy(omega_cuda, omega, width*sizeof(Complex_cuda), cudaMemcpyHostToDevice);

        fft <<<grid, block>>>(h_cuda, omega_cuda, width);
        gpuErrchk(cudaPeekAtLastError());
        gpuErrchk(cudaDeviceSynchronize());
        cudaMemcpy(X, h_cuda, width*height*sizeof(Complex_cuda), cudaMemcpyDeviceToHost);


		chrono::steady_clock::time_point tEnd = chrono::steady_clock::now();
		chrono::duration<double> time_span = chrono::duration_cast<chrono::duration<double>>(tEnd - tStart);
		cout << "Time ellipsed: " << time_span.count() << " seconds... \n";
		cout << "Printing results...\n";
		inImage.save_image_data(outputFile, X, width, height);
        free(h); 
        free(X);
        cudaFree(h_cuda);
	}
	else { // reverse DFT
		Complex_cuda *H = inImage.get_image_data();
        Complex_cuda *h = (Complex_cuda *)malloc(width*height*sizeof(Complex_cuda));
        Complex_cuda *H_cuda;
        Complex_cuda* omega_inverse = (Complex_cuda *)malloc(width*sizeof(Complex_cuda));
        Complex_cuda* omega_inverse_cuda;
        cudaMalloc((void **)&H_cuda, width * height * sizeof(Complex_cuda));
        cudaMalloc((void **)&omega_inverse_cuda, width * sizeof(Complex_cuda));
        
		cout << "Start clokcing...\n";
        chrono::steady_clock::time_point tStart = chrono::steady_clock::now();
        
        int block_size = (height>512)?(512):(height);
        dim3 grid(Divup(height,block_size),1);    
        dim3 block(block_size,1);
        cudaMemcpy(H_cuda, H, width*height*sizeof(Complex_cuda), cudaMemcpyHostToDevice);
        cudaMemcpy(omega_inverse_cuda, omega_inverse, width*sizeof(Complex_cuda), cudaMemcpyHostToDevice);
        
        ifft<<<grid, block>>>(H_cuda, omega_inverse_cuda, width);
      
		cudaMemcpy(h, H_cuda, width*height*sizeof(Complex_cuda), cudaMemcpyDeviceToHost);

		chrono::steady_clock::time_point tEnd = chrono::steady_clock::now();
		chrono::duration<double> time_span = chrono::duration_cast<chrono::duration<double>>(tEnd - tStart);
		cout << "Time ellipsed: " << time_span.count() << " seconds... \n";
		cout << "Printing results...\n";
        inImage.save_image_data_real(outputFile, h, width, height);
        free(H); 
        free(h);
        cudaFree(H_cuda); 
	}

	cout << "finished...\n";
	return 0;
}
