#include <wb.h>


#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));      \
      return -1;                                                          \
    }                                                                     \
  } while (0)

//@@ INSERT CODE HERE


__global__ void RotateParallel(float* Source, float* Destination, int sizeX, int sizeY, float deg,int channel)
{
	//printf("size ====== %d", sizeX*sizeY);
	// x = widght y = height	
	
	int tdx = (blockDim.x*blockIdx.x) + threadIdx.x;
	int layer = threadIdx.y;
	//printf("%f ", Destination[2]);
	
	/*
	printf("************************************************************************************************************   ");
	printf("%f ", Source[0]);
	printf("   ************************************************************************************************************");*/
	//Destination = Source;

	//int i = (blockDim.x*blockIdx.x) + threadIdx.x;

	if (tdx < sizeX*sizeY*channel) {
		int pix = (sizeY*((tdx % sizeX) + 1)) - ((tdx / sizeX) + 1);
		if (tdx == 1) {
			printf("********** tdx = %d : layer = %d : pix = %d **********", tdx, layer, pix);
		}
		
		Destination[pix * 3 + layer] = Source[tdx * 3 + layer];
	}
	
	/*
	
	for(int i = 0; i < sizeX*sizeY; i++) {
		int pix = (sizeY*((i % sizeX) + 1)) - ((i / sizeX)+1);
		if (i < 112) {
			printf("++ i = %d ++++ %d", i, pix);
		}
		
		Destination[pix*3] = Source[i*3];
		Destination[pix*3+1] = Source[i*3+1];
		Destination[pix*3+2] = Source[i*3+2];
	}*/
	/*
	for (int i = 0; i < sizeY * 3; i++) {
		
		
		Source[i] = 50;
	}
	*/
	/*
	printf("************************************************************************************************************   ");
	printf("%f ++++++ Des ++++++ %f ", Source[2],Destination[2]);
	printf("   ************************************************************************************************************");
	*/
	/*
	int i = blockIdx.x * blockDim.x + threadIdx.x;// Kernel definition
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	int xc = sizeX - sizeX / 2;
	int yc = sizeY - sizeY / 2;
	int newx = ((float)i - xc)*cos(deg) - ((float)j - yc)*sin(deg) + xc;
	int newy = ((float)i - xc)*sin(deg) + ((float)j - yc)*cos(deg) + yc;
	__syncthreads();
	if (newx >= 0 && newx < sizeX && newy >= 0 && newy < sizeY)
	{
	putPixVal(Destination, sizeX, i, j, readPixVal(Source, sizeX, newx, newy));
	}*/
	
}

__global__ void Rotate(float* Source, float* Destination, int sizeX, int sizeY, float deg, int channel)
{
	//printf("size ====== %d", sizeX*sizeY);
	// x = widght y = height	

	
	//printf("%f ", Destination[2]);

	/*
	printf("************************************************************************************************************   ");
	printf("%f ", Source[0]);
	printf("   ************************************************************************************************************");*/
	//Destination = Source;

	//int i = (blockDim.x*blockIdx.x) + threadIdx.x;
	/*
	if (tdx < sizeX*sizeY*channel) {
		int pix = (sizeY*((tdx % sizeX) + 1)) - ((tdx / sizeX) + 1);
		if (tdx == 131) {
			printf("********** tdx = %d : layer = %d : pix = %d **********", tdx, layer, pix);
		}

		Destination[pix * 3 + layer] = Source[tdx * 3 + layer];
	}
	*/
	

	for(int i = 0; i < sizeX*sizeY; i++) {
	int pix = (sizeY*((i % sizeX) + 1)) - ((i / sizeX)+1);
	if (i < 112) {
	printf("++ i = %d ++++ %d", i, pix);
	}

	Destination[pix*3] = Source[i*3];
	Destination[pix*3+1] = Source[i*3+1];
	Destination[pix*3+2] = Source[i*3+2];
	}
	/*
	for (int i = 0; i < sizeY * 3; i++) {


	Source[i] = 50;
	}
	*/
	/*
	printf("************************************************************************************************************   ");
	printf("%f ++++++ Des ++++++ %f ", Source[2],Destination[2]);
	printf("   ************************************************************************************************************");
	*/
	/*
	int i = blockIdx.x * blockDim.x + threadIdx.x;// Kernel definition
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	int xc = sizeX - sizeX / 2;
	int yc = sizeY - sizeY / 2;
	int newx = ((float)i - xc)*cos(deg) - ((float)j - yc)*sin(deg) + xc;
	int newy = ((float)i - xc)*sin(deg) + ((float)j - yc)*cos(deg) + yc;
	__syncthreads();
	if (newx >= 0 && newx < sizeX && newy >= 0 && newy < sizeY)
	{
	putPixVal(Destination, sizeX, i, j, readPixVal(Source, sizeX, newx, newy));
	}*/

}


int main(int argc, char *argv[]) {
	wbArg_t args;
	int imageChannels;
	int imageWidth;
	int imageHeight;
	char *inputImageFile;
	wbImage_t inputImage;
	wbImage_t outputImage;
	float *hostInputImageData;
	float *hostOutputImageData;
	float *deviceInputImageData;
	float *deviceOutputImageData;

	args = wbArg_read(argc, argv); /* parse the input arguments */

	inputImageFile = wbArg_getInputFile(args, 0);

	inputImage = wbImport(inputImageFile);

	imageWidth = wbImage_getWidth(inputImage);
	imageHeight = wbImage_getHeight(inputImage);
	// For this lab the value is always 3
	imageChannels = wbImage_getChannels(inputImage);

	// Since the image is monochromatic, it only contains one channel
	outputImage = wbImage_new( imageHeight, imageWidth, imageChannels);
	printf("********************");
	printf("Height %d -------- Width %d ",imageHeight, imageWidth);
	printf("********************");
	hostInputImageData = wbImage_getData(inputImage);
	hostOutputImageData = wbImage_getData(outputImage);

	wbTime_start(GPU, "Doing GPU Computation (memory + compute)");

	wbTime_start(GPU, "Doing GPU memory allocation");
	cudaMalloc((void **)&deviceInputImageData,
		imageWidth * imageHeight * imageChannels * sizeof(float));
	cudaMalloc((void **)&deviceOutputImageData,
		imageWidth * imageHeight * imageChannels *sizeof(float));
	wbTime_stop(GPU, "Doing GPU memory allocation");
	printf(" UUUUUUUUUUUUUUUUUUUUUUUUUUUU ");
	printf("%f ", hostInputImageData[2]);
	wbTime_start(Copy, "Copying data to the GPU");
	cudaMemcpy(deviceInputImageData, hostInputImageData,
		imageWidth * imageHeight * imageChannels * sizeof(float),
		cudaMemcpyHostToDevice);
	cudaMemcpy(deviceOutputImageData, hostOutputImageData,
		imageWidth * imageHeight * imageChannels * sizeof(float),
		cudaMemcpyHostToDevice);
	wbTime_stop(Copy, "Copying data to the GPU");

	///////////////////////////////////////////////////////
	wbTime_start(Compute, "Doing the computation on the GPU");
	//@@ INSERT CODE HERE
	int blockSize = ((imageWidth*imageHeight*imageChannels) / 1023) + 1;

	RotateParallel << < dim3(blockSize,1,1), dim3(341,3,1)>> > (deviceInputImageData, deviceOutputImageData, imageWidth, imageHeight, 90, imageChannels);

	//Rotate <<< 1,1>>> (deviceInputImageData, deviceOutputImageData, imageWidth, imageHeight, 90, imageChannels);
	wbTime_stop(Compute, "Doing the computation on the GPU");

	///////////////////////////////////////////////////////
	wbTime_start(Copy, "Copying data from the GPU");
	cudaMemcpy(hostOutputImageData, deviceOutputImageData,
		imageWidth * imageHeight * imageChannels * sizeof(float),
		cudaMemcpyDeviceToHost);
	wbTime_stop(Copy, "Copying data from the GPU");

	wbTime_stop(GPU, "Doing GPU Computation (memory + compute)");
	printf(" +++++++++++++++++++++++++++++++++    ");
	printf("%f ------- ", hostOutputImageData[2]);
	wbSolution(args, outputImage);

	cudaFree(deviceInputImageData);
	cudaFree(deviceOutputImageData);

	wbImage_delete(outputImage);
	wbImage_delete(inputImage);

	return 0;
}



