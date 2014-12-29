
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cuComplex.h"
#include "math_functions.h"
#include "math_constants.h":
#include <cufft.h>

#include <stdio.h>

cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size);

__global__ void addKernel(int *c, const int *a, const int *b)
{
	int i = threadIdx.x;
	c[i] = a[i] + b[i];
}

__device__ __forceinline__ cuComplex expf(cuComplex z)
{

	cuComplex res;
	float t = expf(z.x);
	sincosf(z.y, &res.y, &res.x);
	res.x *= t;
	res.y *= t;

	return res;

}

__global__ void calculate(cuComplex *fths, int *xo, int *yo, int *uo, float *zo2, float *dfxs, int *Nxs, float *lambda, int *Ts, float * fxs, float* y0seg, float* x0seg, int S_Bx, int S_By, int N_Bx, int N_By)
{
	int	index = blockIdx.y*(blockDim.x*gridDim.x) + blockIdx.x*blockDim.x + threadIdx.x;
	//	int index = threadIdx.x*blockIdx.z + blockIdx.y*blockDim.z + blockIdx.x*blockDim.x;


	//	int blockId = blockIdx.x + blockIdx.y * gridDim.x  + gridDim.x * gridDim.y * blockIdx.z;
	//	int threadId = blockId * blockDim.x + threadIdx.x;

	//	int pt_indx = threadIdx.x*blockIdx.z;

	float yp = yo[threadIdx.x] - y0seg[blockIdx.y];
	float xp = xo[threadIdx.x] - x0seg[blockIdx.x];

	float rp = sqrt(zo2[blockIdx.x] + xp*xp + yp*yp);


	float inv_rp = 1 / rp;

	float fxp = xp*inv_rp / *lambda;
	float fyp = yp*inv_rp / *lambda;

	float k0 = 2 * CUDART_PI_F / *lambda;

	int iifx = round(fxp / *dfxs) + *Nxs / 2 + 1;
	int iify = round(fyp / *dfxs) + *Nxs / 2 + 1;

	if (iifx <= 0 || iifx > *Nxs || iify <= 0 || iify > *Nxs){
		iifx = *Nxs / 2 + 1;
		iify = *Nxs / 2 + 1;
	}

	cuComplex c0;
	cuComplex arg;
	arg.x = (k0*rp - 2 * CUDART_PI_F*(fxs[iifx] + fxs[iify])*(*Ts / 2)*inv_rp);

	c0 = expf(arg);
	c0.x = uo[blockDim.x] * c0.x;
	c0.y = uo[blockDim.x] * c0.y;

	//fths[threadId] = c0;
//	Nep[threadId] = iifx;
//	Nip[threadId] = iify;

//	fths[iifx + blockIdx.x*S_Bx + iify*S_Bx*N_Bx + blockIdx.x* S_Bx*N_Bx*S_By].x += c0.x;

//	fths[iifx + blockIdx.x*S_Bx + iify*S_Bx*N_Bx + blockIdx.x* S_Bx*N_Bx*S_By].y += c0.y;
}

cufftResult preparePlan2D(cufftHandle* plan, int nRows, int nCols, int batch){

	int n[2] = { nRows, nCols };

	cufftResult result = cufftPlanMany(plan,
		2, //rank
		n, //dimensions = {nRows, nCols}
		0, //inembed
		batch, //istride
		1, //idist
		0, //onembed
		batch, //ostride
		1, //odist
		CUFFT_C2C, //cufftType
		batch /*batch*/);

	if (result != 0){

//		std::cout << "preparePlan2D error, result: " << result << "/n";
		return result;
	}
	return result;
}

cufftResult execute2D(cufftHandle* plan, cufftComplex* idata, cufftComplex* odata, int direction){

	cufftResult result = cufftExecC2C(*plan, idata, odata, direction);

	if (result != 0){

//		cout << "execute2D error, result: " << result << "/n";
		return result;
	}
	return result;
}


__global__ void copy2bitmap(cuComplex *H, int *bitmap_H)
{

}


__global__ void asemble(cuComplex *fths, int *xo, int *yo, int *uo, float *zo2, float *dfxs, int *Nxs, float *lambda, int *Ts, float * fxs, float* y0seg, float* x0seg, int* Nep, int* Nip)
{
	//int	index = blockIdx.y*(blockDim.x*gridDim.x) + blockIdx.x*blockDim.x + threadIdx.x;
	//	int index = threadIdx.x*blockIdx.z + blockIdx.y*blockDim.z + blockIdx.x*blockDim.x;


	int blockId = blockIdx.x + blockIdx.y * gridDim.x + gridDim.x * gridDim.y * blockIdx.z;
	int threadId = blockId * blockDim.x + threadIdx.x;

	int pt_indx = threadIdx.x*blockIdx.z;

//	fths2[] = fths2[] + fths[threadId]

//		fths[threadId] = c0;
//	Nep[threadId] = iifx;
//	Nip[threadId] = iify;


}


void CPAS_CGH_3DPS_2d(int* xo, int Np, int* yo, int yo_size, int* zo, int zo_size, int* uo, int Nx, int Ny, int dx, float lambda, int Nxs)
{



	int x_size = (Nx / 2) + ((Nx / 2) - 1) + 1;
	int y_size = (Ny / 2) + ((Ny / 2) - 1) + 1;

	float *x = (float*)malloc(x_size * sizeof(float));
	float *y = (float*)malloc(y_size * sizeof(float));

	for (int t = 0; t < x_size; t++){
		x[t] = (-Nx / 2 + t)*dx;
	}

	for (int t = 0; t < y_size; t++){
		y[t] = (-Ny / 2 + t)*dx;
	}

	int Nosx = Nx / Nxs;
	int Nosy = Ny / Nxs;

	int Ts = Nxs*dx;
	float dfxs = 1 / (float)Ts;

	int fxs_size = (Nxs / 2) + ((Nxs / 2) - 1) + 1;
	float *fxs = (float*)malloc(fxs_size * sizeof(float));

	for (int t = 0; t < fxs_size; t++){
		fxs[t] = (float)(-Nxs / 2 + t)*dfxs;
	}

	float * x0seg = (float*)malloc((Nosx)* sizeof(float));

	for (int t = 0; t < Nosx; t++){
		x0seg[t] = (x[0] + (t*Ts) + Ts / 2);
	}

	float * y0seg = (float*)malloc((Nosy)* sizeof(float));

	for (int t = 0; t < Nosy; t++){
		y0seg[t] = (y[0] + (t*Ts) + Ts / 2);
	}

	float * nseg_bx = (float*)malloc((Nosx)* sizeof(float));

	for (int t = 0; t < Nosx; t++){
		nseg_bx[t] = (1 + (t*Nxs));

	}

	float * nseg_by = (float*)malloc((Nosy)* sizeof(float));

	for (int t = 0; t < Nosy; t++){
		nseg_by[t] = (1 + (t*Nxs));
	}

	float *h = (float*)calloc(Nx, sizeof(float));
	float zo2 = zo[0] * zo[0];

	cudaEvent_t start, stop;
	float time;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);


	//fths[x + (y*Nxs) + (z*Nxs*Nosx)];

	cuComplex *fths;

	cudaMalloc(&fths, sizeof(cuComplex)*Nosx*Nosy*Nxs*Nxs);
	cudaMemset(fths, 0, sizeof(cuComplex)*Nxs*Nxs*Nosx*Nosy);

	dim3 grid;
	grid.x = Nosx;//y
	grid.y = Nosy;//x

	dim3 block;
	block.x = Np; //z
	block.y = 1;
	
	cudaEventRecord(start, 0);
	calculate <<< grid, block >>>(fths, xo, yo, uo, &zo2, &dfxs, &Nxs, &lambda, &Ts, fxs, y0seg, x0seg, 16, 16 ,128 ,128);

	cuComplex *host;
	host = (cuComplex*)malloc(sizeof(cuComplex)*Nosx*Nosy*Np);
	cudaMemcpy(host, fths, sizeof(cuComplex)*Nosx*Nosy*Np, cudaMemcpyDeviceToHost);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&time, start, stop);
	printf("Time for the kernel: %f ms\n", time);

}

int main()
{
	const int arraySize = 5;
	const int a[arraySize] = { 1, 2, 3, 4, 5 };
	const int b[arraySize] = { 10, 20, 30, 40, 50 };
	int c[arraySize] = { 0 };

	cudaEvent_t start, stop;
	float time;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	// Add vectors in parallel.
	cudaError_t cudaStatus = addWithCuda(c, a, b, arraySize);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addWithCuda failed!");
		return 1;
	}

	printf("{1,2,3,4,5} + {10,20,30,40,50} = {%d,%d,%d,%d,%d}\n",
		c[0], c[1], c[2], c[3], c[4]);

	int nRows = 16;
	int nCols = 16;

	cufftComplex* h_in; //dane wejœciowe CPU
	cufftComplex* h_out; //dane wynikowe CPU
	cufftComplex* d_in; //dane wejœciowe GPU
	cufftComplex* d_out; //dane wyjœciowe GPU

	int batch = 128 * 128;
	cufftHandle forwardPlan;


	cudaEventRecord(start, 0);

	preparePlan2D(&forwardPlan, nRows, nCols, batch);

//	h_in = convertMatToCufftComplex(&image, nCols, nRows, false); //konwersja obrazu do typu cufftComplex (CPU)
	h_out = (cufftComplex*)malloc(sizeof(cufftComplex)*nRows*nCols*batch); //allokacja pamiêci na wynik (CPU)
	h_in = (cufftComplex*)malloc(sizeof(cufftComplex)*nRows*nCols*batch); //allokacja pamiêci na wynik (CPU)

	cudaMalloc(&d_in, sizeof(cufftComplex)*nRows*nCols*batch); //allokacja pamiêci na dane wejœciowe (GPU)
		
	cudaEventRecord(start, 0);
		
		cudaMemcpy(d_in, h_in, sizeof(cufftComplex)*nRows*nCols*batch, cudaMemcpyHostToDevice); //kopiowanie danych wejœciowych na GPU

	cudaMalloc(&d_out, sizeof(cufftComplex)*nRows*nCols*batch); //allokacja pamiêci na dane wyjœciowe (GPU)
	cudaMemset(d_out, 0, sizeof(cufftComplex)*nRows*nCols*batch); //Wype³nianie zaalokowanej pamiêci zerami (GPU)

	/*Kod kernela*/
	int xo_size = 10;
	int yo_size = 10;
	int zo_size = 10;
	int uo_size = 10;

	int *xo;
	int *yo;
	int *zo;
	int *uo;

	xo = (int*)malloc((xo_size)* sizeof(int));
	yo = (int*)malloc((yo_size)* sizeof(int));
	zo = (int*)malloc((zo_size)* sizeof(int));
	uo = (int*)malloc((uo_size)* sizeof(int));
	
	int Nx = 2048;
	int Ny = 2048;
	int dx = 8;
	float lambda = 0.5;
	float Nsx = 16;

	CPAS_CGH_3DPS_2d(xo, xo_size, yo, yo_size, zo, zo_size,uo, Nx, Ny, dx,lambda,Nsx);

	/*Koniec*/
cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	execute2D(&forwardPlan, d_in, d_out, CUFFT_FORWARD); //Policzenie FFT
	
	
	
	cudaMemcpy(h_out, d_out, sizeof(cufftComplex)*nRows*nCols*batch, cudaMemcpyDeviceToHost); //Kopiowanie wyniku do pamiêci CPU

	//h_out wynik zawieraj¹cy czêœæ rzeczywist¹ i urojon¹






	// Retrieve result from device and store it in host array
	cudaEventElapsedTime(&time, start, stop);
	printf("Time for the kernel: %f ms\n", time);

	// cudaDeviceReset must be called before exiting in order for profiling and
	// tracing tools such as Nsight and Visual Profiler to show complete traces.
	cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceReset failed!");
		return 1;
	}

	return 0;
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size)
{
	int *dev_a = 0;
	int *dev_b = 0;
	int *dev_c = 0;
	cudaError_t cudaStatus;

	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		goto Error;
	}

	// Allocate GPU buffers for three vectors (two input, one output)    .
	cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	// Copy input vectors from host memory to GPU buffers.
	cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	// Launch a kernel on the GPU with one thread for each element.
	addKernel << <1, size >> >(dev_c, dev_a, dev_b);

	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
		goto Error;
	}

	// Copy output vector from GPU buffer to host memory.
	cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

Error:
	cudaFree(dev_c);
	cudaFree(dev_a);
	cudaFree(dev_b);

	return cudaStatus;
}
