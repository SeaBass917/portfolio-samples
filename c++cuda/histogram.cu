/////////////////////////////////////////////////////////////
// Author(s):     Sebastian Thiem, Ali Akoglu
// Date Created:  24 March 2019
//
// Project:       Module 5 - Histograms
// Filename:      histogram.cu
//
// Description:   3 different versions of a parallelized 
//                histogram:
//                   V1 - global memory only
//                   V2 - shared memory with privatization
//                   V3 - direct to shared streaming
//
/////////////////////////////////////////////////////////////
#include <wb.h>

#define NUM_BINS 4096
#define BLOCK_WIDTH 1024
#define ITTERATIONS 16

#define CUDA_CHECK(ans)                                                   \
   { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line,
                                 bool abort = true) {
   if (code != cudaSuccess) {
      fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code),
                  file, line);
      if (abort)
         exit(code);
   }
}

// version 1: global memory only coalesced memory access 
//    with interleaved partitioning (striding)
__global__ void histogram_kernel(unsigned *input, unsigned *bins, unsigned num_elements, unsigned num_bins) {

   // Where are we in respect to the grid
   unsigned myId = threadIdx.x + blockIdx.x * blockDim.x;

   // Size of grid
   unsigned grid_size = blockDim.x * gridDim.x;
   
   // used to index by stride, start at myid
   unsigned idx = myId;

   // Initialize the bins to 0
   while(idx < NUM_BINS){
      bins[idx] = 0;
      idx+=grid_size;
   }
   __syncthreads();

   // Local variable for the input array read
   unsigned in_loc = 0;

   idx = myId;

   // for threads within the input array
   // loop through each grid
   while(idx < num_elements){

      // read from array
      in_loc = input[idx];

      // If that value is valid add it to bin
      if(in_loc < NUM_BINS){
         atomicAdd(&(bins[in_loc]), 1);
      }

      idx+=grid_size;
   }
}  

// version 2: shared memory with privatization
__global__ void histogram_private_kernel(unsigned *input, unsigned *bins, unsigned num_elements, unsigned num_bins){

   // Shared memory boxes
	__shared__ unsigned int boxs[NUM_BINS];

   // size of grid
   unsigned grid_size = blockDim.x*gridDim.x;

   // Position in grid
   int myId = threadIdx.x + blockIdx.x*blockDim.x;

   // Init shared bins using blocks
   for(int j=threadIdx.x; j<num_bins;j+=blockDim.x){
      boxs[j]=0;
   }
   __syncthreads();

   // reinitialize the index
   unsigned i = myId;

   // execute if we are within the input array
	if(i < num_elements){

      // for each of the elements strided by grid size
		while(i < num_elements){
			atomicAdd( &(boxs[input[i]]),1);
			i+=grid_size;
		}   
	}
   __syncthreads();

   // Write back to global
   for(int j=threadIdx.x; j<num_bins; j+=blockDim.x){
      atomicAdd(&(bins[j]), boxs[j]);
   }
}

// version 3: direct to shared streaming
__global__ void histogram_dir2shr_kernel(unsigned *input, unsigned *bins, unsigned num_elements, unsigned num_bins){

   // Shared memory boxes
	__shared__ unsigned int boxs[NUM_BINS];

   // size of grid
   unsigned grid_size = blockDim.x*gridDim.x;

   // Position in grid
   int myId = threadIdx.x + blockIdx.x*blockDim.x;

   // Init shared bins using blocks
   for(int j=threadIdx.x; j<num_bins;j+=blockDim.x){
      boxs[j]=0;
   }
   __syncthreads();

   // reinitialize the index
   unsigned i = myId;

   // execute if we are within the input array
	if(i < num_elements){

      // for each of the elements strided by grid size
		while(i < num_elements){
			atomicAdd( &(boxs[input[i]]),1);
			i+=grid_size;
		}   
	}
   __syncthreads();

   // Write back to global
   for(int j=threadIdx.x; j<num_bins; j+=blockDim.x){
      atomicAdd(&(bins[j]), boxs[j]);
   }
}

// Saturate the graph to cap at 127
__global__ void saturate_kernel(unsigned *bins, unsigned num_bins){

   // Where are we in respect to the grid
   unsigned myId = threadIdx.x + blockIdx.x * blockDim.x;

   // Check all the bins for oversaturation
   if(myId < NUM_BINS){
      if(bins[myId] > 127){
         bins[myId] = 127;
      }
   }
}

// Top level function to handle the differernt kernel launched and version control
void histogram(unsigned* input, unsigned* bins, unsigned num_elements, unsigned num_bins, unsigned version){

    // Initialize the grid and block dimensions for 256 width thread blocks
    dim3 dimGrid_hist((num_elements - 1)/(BLOCK_WIDTH*ITTERATIONS)+1, 1, 1);
    dim3 dimBlock_hist(BLOCK_WIDTH,1,1);

    // Initialize the grid and block dimensions for 256 width thread blocks
    dim3 dimGrid_sat((NUM_BINS - 1)/BLOCK_WIDTH+1, 1, 1);
    dim3 dimBlock_sat(BLOCK_WIDTH,1,1);

	// Launch different kernels depending on version

	// Global history kernel
	if(version == 0){

		histogram_kernel<<<dimGrid_hist, dimBlock_hist>>>(input, bins, num_elements, NUM_BINS);

	}
	// shared memory privatized version
	else if(version == 1){

		histogram_private_kernel<<<dimGrid_hist, dimBlock_hist>>>(input, bins, num_elements, NUM_BINS);

	}
	else{
		// throw exception
	}


	// Saturate bin results at 127
	saturate_kernel<<<dimGrid_sat, dimBlock_sat>>>(bins, NUM_BINS);
}

int main(int argc, char *argv[]) {

   // command line arguments from library
   wbArg_t args;

   // Host/Device memory with input length variable
   int inputLength;
   unsigned int *h_Input;
   unsigned int *h_Bins;
   unsigned int *hp_Input; // pinned
   unsigned int *hp_Bins;  // pinned
   unsigned int *d_Input;
   unsigned int *d_Bins;

   // Used to hold the memory space required for Input and Bin
   unsigned input_size = 0;
   unsigned bin_size = 0;

   // Read in the command args
   args = wbArg_read(argc, argv);

   // Import the testing data
   wbTime_start(Generic, "Importing data and creating memory on host");
   h_Input = (unsigned int *)wbImport(wbArg_getInputFile(args, 0), &inputLength, "Integer");
   h_Bins = (unsigned int *)malloc(NUM_BINS * sizeof(unsigned int));
   wbTime_stop(Generic, "Importing data and creating memory on host");

   wbLog(TRACE, "The input length is ", inputLength);
   wbLog(TRACE, "The number of bins is ", NUM_BINS);

   // Calculate memory sizes
   input_size = sizeof(unsigned) * inputLength;
   bin_size = sizeof(unsigned) * NUM_BINS;

   // Allocate pinned memory for host
   CUDA_CHECK(cudaMallocHost((void**)&hp_Input, input_size) ); // host pinned
   CUDA_CHECK(cudaMallocHost((void**)&hp_Bins, bin_size) ); // host pinned

   // move input data to pinned memory
   memcpy(hp_Input, h_Input, input_size);

   // Allocate memory
   wbTime_start(GPU, "Allocating GPU memory.");
   cudaMalloc((void**) &d_Input, input_size);
   cudaMalloc((void**) &d_Bins, bin_size);
   CUDA_CHECK(cudaDeviceSynchronize());
   wbTime_stop(GPU, "Allocating GPU memory.");

   // Move memory from host to device
   wbTime_start(GPU, "Copying input memory to the GPU.");
   cudaMemcpy(d_Input, hp_Input, input_size, cudaMemcpyHostToDevice);
   CUDA_CHECK(cudaDeviceSynchronize());
   wbTime_stop(GPU, "Copying input memory to the GPU.");

   // Launch kernel
   wbLog(TRACE, "Launching kernel");
   wbTime_start(Compute, "Performing CUDA computation");
   histogram(d_Input, d_Bins, inputLength, NUM_BINS, 1);
   wbTime_stop(Compute, "Performing CUDA computation");

   // Return resulting memory from the GPU
   wbTime_start(Copy, "Copying output memory to the CPU");
   cudaMemcpy(hp_Bins, d_Bins, bin_size, cudaMemcpyDeviceToHost);
   memcpy(h_Bins, hp_Bins, bin_size);
   CUDA_CHECK(cudaDeviceSynchronize());
   wbTime_stop(Copy, "Copying output memory to the CPU");

   wbTime_start(GPU, "Freeing GPU Memory");
   cudaFree(d_Input);
   cudaFree(d_Bins);
   wbTime_stop(GPU, "Freeing GPU Memory");

   // Verify correctness
   wbSolution(args, h_Bins, NUM_BINS);

   // Free host memory
   cudaFreeHost(hp_Input);
   cudaFreeHost(hp_Bins);
   free(h_Input);
   free(h_Bins);

   return 0;
}
