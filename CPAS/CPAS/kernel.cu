
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

__global__ void calculate(cuComplex *fths, int *xo, int *yo, int *uo, float *zo2, float dfxs,  float lambda, float k0, int Ts, float *fxs, float * y0seg, float* x0seg, int S_Bx, int S_By, int N_Bx, int N_By)
{
	
//	int	index = blockIdx.y*(blockDim.x*gridDim.x) + blockIdx.x*blockDim.x + threadIdx.x;
	//	int index = threadIdx.x*blockIdx.z + blockIdx.y*blockDim.z + blockIdx.x*blockDim.x;


	//	int blockId = blockIdx.x + blockIdx.y * gridDim.x  + gridDim.x * gridDim.y * blockIdx.z;
	//	int threadId = blockId * blockDim.x + threadIdx.x;

	//	int pt_indx = threadIdx.x*blockIdx.z;


	float yp = yo[threadIdx.x] - y0seg[blockIdx.y];
	
	float xp = xo[threadIdx.x] - x0seg[blockIdx.x];

	float rp = sqrt(zo2[threadIdx.x ] + xp*xp + yp*yp);
	

	float inv_rp = 1 / rp;

	float fxp = xp*inv_rp / lambda;
	float fyp = yp*inv_rp / lambda;



	int iifx = round(fxp / dfxs) + S_Bx / 2 + 1;
	int iify = round(fyp / dfxs) + S_By / 2 + 1;

	if (iifx <= 0 || iifx > S_Bx || iify <= 0 || iify > S_Bx){
		iifx = S_Bx / 2 + 1;
		iify = S_Bx / 2 + 1;
	}
	


	cuComplex c0;
	cuComplex arg;
	arg.x = (k0*rp - 2 * CUDART_PI_F*(fxs[iifx] + fxs[iify])*(Ts / 2)*inv_rp);
	
	c0 = expf(arg);
//	c0.x = uo[blockDim.x] * c0.x;
//	c0.y = uo[blockDim.x] * c0.y;

	//fths[threadId] = c0;
//	Nep[threadId] = iifx;
//	Nip[threadId] = iify;

	fths[iifx + blockIdx.x*S_Bx + iify*S_Bx*N_Bx + blockIdx.x* S_Bx*N_Bx*S_By].x += c0.x;
	fths[iifx + blockIdx.x*S_Bx + iify*S_Bx*N_Bx + blockIdx.x* S_Bx*N_Bx*S_By].y += c0.y;
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


void CPAS_CGH_3DPS_2d(int Np, int* xo, int* yo, int* zo, int* uo, int Nx, int Ny, int dx, float lambda, int S_Bx, int S_By, cuComplex* fths_p)
{
	cudaEvent_t start, stop;
	float time;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	double k0 = 2 * CUDART_PI_F / lambda;

	int x_size = (Nx / 2) + ((Nx / 2) - 1) + 1;
	int y_size = (Ny / 2) + ((Ny / 2) - 1) + 1;

	float *x = (float*) malloc(x_size * sizeof(float));
	float *y = (float*) malloc(y_size * sizeof(float));

	for (int t = 0; t < x_size; t++){
		x[t] = (-Nx / 2 + t)*dx;
	}

	for (int t = 0; t < y_size; t++){
		y[t] = (-Ny / 2 + t)*dx;
	}

	int N_Bx = Nx / S_Bx; // dodaæ obs³ugê nie ca³kowitych dzieleñ
	int N_By = Ny / S_By;

	int Ts = S_Bx*dx;
	
	float dfxs = 1 / (float)Ts;

	int fxs_size = (S_Bx / 2) + ((S_Bx / 2) - 1) + 1;
	float *fxs = (float*)malloc(fxs_size * sizeof(float));

	for (int t = 0; t < fxs_size; t++){
		fxs[t] = (float)(-S_Bx / 2 + t)*dfxs;
	}

	float * x0seg = (float*)malloc((N_Bx)* sizeof(float));

	for (int t = 0; t < N_By; t++){
		x0seg[t] = (x[0] + (t*Ts) + Ts / 2);
	}

	float * y0seg = (float*)malloc((N_By)* sizeof(float));

	for (int t = 0; t < N_By; t++){
		y0seg[t] = (y[0] + (t*Ts) + Ts / 2);
	}
/*	float * nseg_bx = (float*)malloc((Nosx)* sizeof(float));

	for (int t = 0; t < Nosx; t++){
		nseg_bx[t] = (1 + (t*Nxs));

	}

	float * nseg_by = (float*)malloc((Nosy)* sizeof(float));

	for (int t = 0; t < Nosy; t++){
		nseg_by[t] = (1 + (t*Nxs));
	}

	float *h = (float*)calloc(Nx, sizeof(float));
*/

	float *z02;
	z02 = (float*)malloc((Np)* sizeof(float));

	for (int t = 0; t < Np; t++)
		z02[t] = zo[t] * zo[t];



//	cudaMalloc(&fths_p, sizeof(cuComplex)*N_Bx*N_By*S_Bx*S_By);
//	cudaMemset(fths_p, 0, sizeof(cuComplex)*N_Bx*N_By*S_Bx*S_By);


	int *d_xo;
	int *d_yo;
	float *d_z02;
	
	cudaMalloc((void**)&d_xo, sizeof(int)*Np);
	cudaMalloc((void**)&d_yo, sizeof(int)*Np);
	cudaMalloc((void**)&d_z02, sizeof(float)*Np);

	cudaMemcpy(d_xo, xo, Np*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_yo, yo, Np*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_z02, z02, Np*sizeof(float), cudaMemcpyHostToDevice);

	float *d_fxs;
	float *d_y0seg;
	float *d_x0seg;

	cudaMalloc((void**)&d_x0seg, sizeof(float)*N_Bx);
	cudaMalloc((void**)&d_y0seg, sizeof(float)*N_By);
	cudaMalloc((void**)&d_fxs, sizeof(float)*fxs_size);

	cudaMemcpy(d_fxs, fxs, fxs_size*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_x0seg, x0seg, N_Bx*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_y0seg, y0seg, N_By*sizeof(float), cudaMemcpyHostToDevice);

	dim3 grid;
	grid.x = N_Bx;//y
	grid.y = N_By;//x

	dim3 block;
	block.x = Np; //z
	block.y = 1;
	
	cudaEventRecord(start, 0);
	calculate << < grid, block >> >(fths_p, d_xo, d_yo, uo, d_z02, dfxs, lambda, k0, Ts, d_fxs, d_y0seg, d_x0seg, S_Bx, S_Bx, N_Bx, N_By);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&time, start, stop);
	printf("Time for the kernel: %f ms\n", time);

/*	cuComplex *host;
	host = (cuComplex*)malloc(sizeof(cuComplex)*Nosx*Nosy*Np);
	cudaMemcpy(host, fths, sizeof(cuComplex)*Nosx*Nosy*Np, cudaMemcpyDeviceToHost);
*/
}

int main()
{




	cudaEvent_t start, stop;
	float time;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	/*START CUDA CALC PART - DEKLARACJE*/
	int Nx = 1024;
	int Ny = 1024;
	int dx = 8;
	float lambda = 0.5;
	/*START CUDA FFT 2D PART - DEKLARACJE*/
	int S_Bx = 8; 
	int S_By = 8;

	cufftComplex* h_out; //dane wynikowe CPU
	cufftComplex* holo; //dane wyjœciowe GPU

	int batch = Nx/S_Bx * Ny/S_By;  //N_Bx*N_By
	cufftHandle forwardPlan;

	preparePlan2D(&forwardPlan, S_Bx, S_By, batch);
	


	h_out = (cufftComplex*)malloc(sizeof(cufftComplex)*S_Bx*S_By*batch); //allokacja pamiêci na wynik (CPU)


	
	cudaMalloc(&holo, sizeof(cufftComplex) *S_Bx*S_By*batch); //allokacja pamiêci na dane wyjœciowe (GPU)
	cudaMemset(holo, 0, sizeof(cufftComplex)*S_Bx*S_By*batch); //Wype³nianie zaalokowanej pamiêci zerami (GPU)

	/*END CUDA FFT 2D PART - DEKLARACJE*/

	/*Kod kernela*/
	int Np = 1000;

	int *xo;
	int *yo;
	int *zo;
	int *uo;
	

	xo = (int*)malloc((Np)* sizeof(int));
	yo = (int*)malloc((Np)* sizeof(int));
	zo = (int*)malloc((Np)* sizeof(int));
	uo = (int*)malloc((Np)* sizeof(int));

	for (int tt = 0; tt < Np; tt++)
	{
		xo[tt] = tt;
		yo[tt] = Np - tt;
		zo[tt] = yo[tt] * xo[tt];
	}


	
	cuComplex *fths_p;

	cufftComplex* fhs; 

//	cudaMalloc(&fhs, sizeof(cufftComplex)*S_Bx*S_By*batch); //allokacja pamiêci na dane wejœciowe (GPU)
	cudaMalloc(&fths_p, sizeof(cuComplex)*Nx*Ny);
	cudaMemset(fths_p, 0, sizeof(cuComplex)*Nx*Ny);

	cudaEventRecord(start, 0);
	/*START CUDA CALC PART */
	CPAS_CGH_3DPS_2d(Np, xo, yo, zo, uo, Nx, Ny, dx, lambda, S_Bx, S_By, fths_p);

	/*START CUDA CALC PART */

	
	


	/*START CUDA FFT PART */
	execute2D(&forwardPlan, fths_p, holo, CUFFT_FORWARD); 

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);

	/*Wyswietlanie modulu/fazy*/
	cudaMemcpy(h_out, holo, sizeof(cufftComplex)*S_Bx*S_By*batch, cudaMemcpyDeviceToHost); 

	
	/*END CUDA FFT PART */





	// Retrieve result from device and store it in host array
	cudaEventElapsedTime(&time, start, stop);
	printf("Time for the kernel: %f ms\n", time);
	printf("Time for the kernel: %f ms\n", time);

	// cudaDeviceReset must be called before exiting in order for profiling and
	// tracing tools such as Nsight and Visual Profiler to show complete traces.
/*	cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceReset failed!");
		return 1;
	}
	*/
	return 0;
}

