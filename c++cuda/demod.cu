////////////////////////////////////////////////////////////////////////////////////
// Engineer(s):     Kray Althaus, Catherine McIntosh, Josue Ortiz, 
//						Kevin Siruno, Sebastian Thiem
//
// Sponsor:         General Dynamics
// 
// Project Name:    GPU Modulation
// File Name:		demod.cpp
// Create Date:     24 January 2019
//
// Description:     
//
////////////////////////////////////////////////////////////////////////////////////
#include "demod.cuh"
#include "srrc.cuh"
#include "util.cuh"
#include <stdio.h>
#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include <iterator>
#include <math.h>

// convert 8 padded 3-bit symbols to 3 compact bytes
void glue(char* comp, char* padd) {
   comp[0] = (padd[0] << 5) | (padd[1] << 2) | (padd[2] >> 1);
   comp[1] = (padd[2] << 7) | (padd[3] << 4) | (padd[4] << 1) | (padd[5] >> 2);
   comp[2] = (padd[5] << 6) | (padd[6] << 3) | (padd[7]);
}

// convert upsampled boi to downsampled boi
void downsample_0(float* data_ds, float* data, unsigned len_ds) {

   // 1 thread per upsample rate symbols
   // if we mapped the threads 1-1 wed have divergence
   for (unsigned id = 0; id < len_ds; id++) {

      // DSRATE * 2 cuz 2 floats per symbol
      unsigned idx = id * DSRATE * 2;
      unsigned idx_max = 0;
      float val_max = 0;
      float val_cur = 0;
      float I = 0;
      float Q = 0;

      // for each sample in the DSRATE
      // find the sample with the highest energy
      for (unsigned i = 0; i < DSRATE * 2; i += 2) {

         I = data[idx + i];
         Q = data[idx + i + 1];

         val_cur = I * I + Q * Q;   // this'll be our energy, skipping the sqrt()

         // Update the max
         if (val_cur > val_max) {
            val_max = val_cur;
            idx_max = i;
         }
      }

      // this is the highest energy symbol
      I = data[idx + idx_max];
      Q = data[idx + idx_max + 1];

      // write it to the downsampled array
      data_ds[id * 2] = I;
      data_ds[id * 2 + 1] = Q;
   }
}

// Analyzes upsampled chunks of the data
// write the symbol in that chunk with the highest energy to the downsampled array
// TODO: Check for register usage, there are less readable ways to write this with fewer registers
__global__ void downsample_kernel_0(float* data_ds, float* data, unsigned len_ds) {

   // Id of the thread with respect to the grid
   unsigned id = threadIdx.x + blockIdx.x * blockDim.x;

   // 1 thread per upsample rate symbols
   // if we mapped the threads 1-1 wed have divergence
   if (id < len_ds) {

      // DSRATE * 2 cuz 2 floats per symbol
      unsigned idx = id * DSRATE * 2;
      unsigned idx_max = 0;
      float val_max = 0;
      float val_cur = 0;
      float I = 0;
      float Q = 0;

      // for each sample in the DSRATE
      // find the sample with the highest energy
      for (unsigned i = 0; i < DSRATE * 2; i += 2) {

         I = data[idx + i];
         Q = data[idx + i + 1];

         val_cur = I * I + Q * Q;   // this'll be our energy, skipping the sqrt()

         // Update the max
         if (val_cur > val_max) {
            val_max = val_cur;
            idx_max = i;
         }
      }

      // this is the highest energy symbol
      I = data[idx + idx_max];
      Q = data[idx + idx_max + 1];

      // write it to the downsampled array
      data_ds[id * 2] = I;
      data_ds[id * 2 + 1] = Q;

   }
}

void downsample(float* data_ds, float* data, unsigned len_ds) {

   unsigned idx = 0;

   // 1 thread per sample rate symbols
   for (unsigned id = 0; id < 2 * len_ds; id+=2) {

      // 2 floats per symbol DSRATE symbols per sample
      idx = DSRATE * id;

      // write it to the downsampled array
      data_ds[id] = data[idx];
      data_ds[id + 1] = data[idx + 1];
   }
}

// ths kernel assumes we know the offset we just need to move memory
__global__ void downsample_kernel(float* data_ds, float* data, unsigned len_ds) {

   // Id of the thread with respect to the grid
   unsigned id = threadIdx.x + blockIdx.x * blockDim.x;

   // 1 thread per sample rate symbols
   if (id < len_ds) {

      id *= 2;

      // 2 floats per symbol DSRATE symbols per sample
      unsigned idx = DSRATE * id;

      // write it to the downsampled array
      data_ds[id] = data[idx];
      data_ds[id + 1] = data[idx + 1];

   }
}

void decoder(char* messageDecoded, float* data_ds, unsigned len_ds) {

   for (unsigned id = 0; id < len_ds / 3; id++) {

      float symbols[16];
      float thetas[8];
      char c[8];

      // 8 inputs & 3 outputs per thread
      unsigned idx_r = id * 16;
      unsigned idx_w = id * 3;

      // Read in the input symbols
      symbols[0] = data_ds[idx_r];
      symbols[1] = data_ds[idx_r + 1];
      symbols[2] = data_ds[idx_r + 2];
      symbols[3] = data_ds[idx_r + 3];
      symbols[4] = data_ds[idx_r + 4];
      symbols[5] = data_ds[idx_r + 5];
      symbols[6] = data_ds[idx_r + 6];
      symbols[7] = data_ds[idx_r + 7];
      symbols[8] = data_ds[idx_r + 8];
      symbols[9] = data_ds[idx_r + 9];
      symbols[10] = data_ds[idx_r + 10];
      symbols[11] = data_ds[idx_r + 11];
      symbols[12] = data_ds[idx_r + 12];
      symbols[13] = data_ds[idx_r + 13];
      symbols[14] = data_ds[idx_r + 14];
      symbols[15] = data_ds[idx_r + 15];

      // Map the angle to a value between 0-7
      thetas[0] = atan2f(symbols[1], symbols[0]);
      thetas[1] = atan2f(symbols[3], symbols[2]);
      thetas[2] = atan2f(symbols[5], symbols[4]);
      thetas[3] = atan2f(symbols[7], symbols[6]);
      thetas[4] = atan2f(symbols[9], symbols[8]);
      thetas[5] = atan2f(symbols[11], symbols[10]);
      thetas[6] = atan2f(symbols[13], symbols[12]);
      thetas[7] = atan2f(symbols[15], symbols[14]);

      // add pi/8 and an additional 2*pi if the angle is negative
      thetas[0] += PI / 8;
      thetas[1] += PI / 8;
      thetas[2] += PI / 8;
      thetas[3] += PI / 8;
      thetas[4] += PI / 8;
      thetas[5] += PI / 8;
      thetas[6] += PI / 8;
      thetas[7] += PI / 8;

      thetas[0] += (thetas[0] < 0) ? 2 * PI : 0;
      thetas[1] += (thetas[1] < 0) ? 2 * PI : 0;
      thetas[2] += (thetas[2] < 0) ? 2 * PI : 0;
      thetas[3] += (thetas[3] < 0) ? 2 * PI : 0;
      thetas[4] += (thetas[4] < 0) ? 2 * PI : 0;
      thetas[5] += (thetas[5] < 0) ? 2 * PI : 0;
      thetas[6] += (thetas[6] < 0) ? 2 * PI : 0;
      thetas[7] += (thetas[7] < 0) ? 2 * PI : 0;

      // Map the angle to a value between 0-7 with (char)(theta * 8 / (2*PI));
      c[0] = (char)(thetas[0] * 4 / PI);
      c[1] = (char)(thetas[1] * 4 / PI);
      c[2] = (char)(thetas[2] * 4 / PI);
      c[3] = (char)(thetas[3] * 4 / PI);
      c[4] = (char)(thetas[4] * 4 / PI);
      c[5] = (char)(thetas[5] * 4 / PI);
      c[6] = (char)(thetas[6] * 4 / PI);
      c[7] = (char)(thetas[7] * 4 / PI);

      // gray code 
      c[0] = c[0] ^ (c[0] >> 1);
      c[1] = c[1] ^ (c[1] >> 1);
      c[2] = c[2] ^ (c[2] >> 1);
      c[3] = c[3] ^ (c[3] >> 1);
      c[4] = c[4] ^ (c[4] >> 1);
      c[5] = c[5] ^ (c[5] >> 1);
      c[6] = c[6] ^ (c[6] >> 1);
      c[7] = c[7] ^ (c[7] >> 1);

      // write the back to message glued
      messageDecoded[idx_w] = (c[0] << 5) | (c[1] << 2) | (c[2] >> 1);
      messageDecoded[idx_w + 1] = (c[2] << 7) | (c[3] << 4) | (c[4] << 1) | (c[5] >> 2);
      messageDecoded[idx_w + 2] = (c[5] << 6) | (c[6] << 3) | (c[7]);
   }
}

void decoder(char* messageDecoded, float* data_ds, unsigned len_ds, const float offset) {

   for (unsigned id = 0; id < len_ds / 3; id++) {

      float I, Q;
      float symbols[16];
      float thetas[8];
      char c[8];

      // 8 inputs & 3 outputs per thread
      unsigned idx_r = id * 16;
      unsigned idx_w = id * 3;

      // Read in the input symbols
      symbols[0] = data_ds[idx_r];
      symbols[1] = data_ds[idx_r + 1];
      symbols[2] = data_ds[idx_r + 2];
      symbols[3] = data_ds[idx_r + 3];
      symbols[4] = data_ds[idx_r + 4];
      symbols[5] = data_ds[idx_r + 5];
      symbols[6] = data_ds[idx_r + 6];
      symbols[7] = data_ds[idx_r + 7];
      symbols[8] = data_ds[idx_r + 8];
      symbols[9] = data_ds[idx_r + 9];
      symbols[10] = data_ds[idx_r + 10];
      symbols[11] = data_ds[idx_r + 11];
      symbols[12] = data_ds[idx_r + 12];
      symbols[13] = data_ds[idx_r + 13];
      symbols[14] = data_ds[idx_r + 14];
      symbols[15] = data_ds[idx_r + 15];

      // Adjust phase offset in here
      I = symbols[0] * cos(offset) - symbols[1] * sin(offset);
      Q = symbols[0] * sin(offset) + symbols[1] * cos(offset);
      symbols[0] = I;
      symbols[1] = Q;
      I = symbols[2] * cos(offset) - symbols[3] * sin(offset);
      Q = symbols[2] * sin(offset) + symbols[3] * cos(offset);
      symbols[2] = I;
      symbols[3] = Q;
      I = symbols[4] * cos(offset) - symbols[5] * sin(offset);
      Q = symbols[4] * sin(offset) + symbols[5] * cos(offset);
      symbols[4] = I;
      symbols[5] = Q;
      I = symbols[6] * cos(offset) - symbols[7] * sin(offset);
      Q = symbols[6] * sin(offset) + symbols[7] * cos(offset);
      symbols[6] = I;
      symbols[7] = Q;
      I = symbols[8] * cos(offset) - symbols[9] * sin(offset);
      Q = symbols[8] * sin(offset) + symbols[9] * cos(offset);
      symbols[8] = I;
      symbols[9] = Q;
      I = symbols[10] * cos(offset) - symbols[11] * sin(offset);
      Q = symbols[10] * sin(offset) + symbols[11] * cos(offset);
      symbols[10] = I;
      symbols[11] = Q;
      I = symbols[12] * cos(offset) - symbols[13] * sin(offset);
      Q = symbols[12] * sin(offset) + symbols[13] * cos(offset);
      symbols[12] = I;
      symbols[13] = Q;
      I = symbols[14] * cos(offset) - symbols[15] * sin(offset);
      Q = symbols[14] * sin(offset) + symbols[15] * cos(offset);
      symbols[14] = I;
      symbols[15] = Q;

      // Map the angle to a value between 0-7
      thetas[0] = atan2f(symbols[1], symbols[0]);
      thetas[1] = atan2f(symbols[3], symbols[2]);
      thetas[2] = atan2f(symbols[5], symbols[4]);
      thetas[3] = atan2f(symbols[7], symbols[6]);
      thetas[4] = atan2f(symbols[9], symbols[8]);
      thetas[5] = atan2f(symbols[11], symbols[10]);
      thetas[6] = atan2f(symbols[13], symbols[12]);
      thetas[7] = atan2f(symbols[15], symbols[14]);

      // add pi/8 and an additional 2*pi if the angle is negative
		// add pi/8 and an additional 2*pi if the angle is negative
		thetas[0] += PI / 8;
		thetas[1] += PI / 8;
		thetas[2] += PI / 8;
		thetas[3] += PI / 8;
		thetas[4] += PI / 8;
		thetas[5] += PI / 8;
		thetas[6] += PI / 8;
		thetas[7] += PI / 8;

		thetas[0] += (thetas[0] < 0) ? 2 * PI : 0;
		thetas[1] += (thetas[1] < 0) ? 2 * PI : 0;
		thetas[2] += (thetas[2] < 0) ? 2 * PI : 0;
		thetas[3] += (thetas[3] < 0) ? 2 * PI : 0;
		thetas[4] += (thetas[4] < 0) ? 2 * PI : 0;
		thetas[5] += (thetas[5] < 0) ? 2 * PI : 0;
		thetas[6] += (thetas[6] < 0) ? 2 * PI : 0;
		thetas[7] += (thetas[7] < 0) ? 2 * PI : 0;

      // Map the angle to a value between 0-7 with (char)(theta * 8 / (2*PI));
      c[0] = (char)(thetas[0] * 4 / PI);
      c[1] = (char)(thetas[1] * 4 / PI);
      c[2] = (char)(thetas[2] * 4 / PI);
      c[3] = (char)(thetas[3] * 4 / PI);
      c[4] = (char)(thetas[4] * 4 / PI);
      c[5] = (char)(thetas[5] * 4 / PI);
      c[6] = (char)(thetas[6] * 4 / PI);
      c[7] = (char)(thetas[7] * 4 / PI);

      // gray code 
      c[0] = c[0] ^ (c[0] >> 1);
      c[1] = c[1] ^ (c[1] >> 1);
      c[2] = c[2] ^ (c[2] >> 1);
      c[3] = c[3] ^ (c[3] >> 1);
      c[4] = c[4] ^ (c[4] >> 1);
      c[5] = c[5] ^ (c[5] >> 1);
      c[6] = c[6] ^ (c[6] >> 1);
      c[7] = c[7] ^ (c[7] >> 1);

      // write the back to message glued
      messageDecoded[idx_w] = (c[0] << 5) | (c[1] << 2) | (c[2] >> 1);
      messageDecoded[idx_w + 1] = (c[2] << 7) | (c[3] << 4) | (c[4] << 1) | (c[5] >> 2);
      messageDecoded[idx_w + 2] = (c[5] << 6) | (c[6] << 3) | (c[7]);
   }
}

__global__ void decoder_kernel(char* messageDecoded, float* data_ds, unsigned len_ds, const float offset) {
   
   // Id of the thread with respect to the grid
   unsigned id = threadIdx.x + blockIdx.x * blockDim.x;

   // every 1 thread will 
   // read from 8 input addresses
   // write to 3 output addresses
   if (id < len_ds / 3) {

      float I, Q; // temp vars used for the phase adjust

      float symbols[16];
      float thetas[8];
      char c[8];

      // 8 inputs & 3 outputs per thread
      unsigned idx_r = id * 16;
      unsigned idx_w = id * 3;

      // Read in the input symbols
      symbols[0] = data_ds[idx_r];
      symbols[1] = data_ds[idx_r + 1];
      symbols[2] = data_ds[idx_r + 2];
      symbols[3] = data_ds[idx_r + 3];
      symbols[4] = data_ds[idx_r + 4];
      symbols[5] = data_ds[idx_r + 5];
      symbols[6] = data_ds[idx_r + 6];
      symbols[7] = data_ds[idx_r + 7];
      symbols[8] = data_ds[idx_r + 8];
      symbols[9] = data_ds[idx_r + 9];
      symbols[10] = data_ds[idx_r + 10];
      symbols[11] = data_ds[idx_r + 11];
      symbols[12] = data_ds[idx_r + 12];
      symbols[13] = data_ds[idx_r + 13];
      symbols[14] = data_ds[idx_r + 14];
      symbols[15] = data_ds[idx_r + 15]; 
      
      // Adjust phase offset in here
      I = symbols[0] * cos(offset) - symbols[1] * sin(offset);
      Q = symbols[0] * sin(offset) + symbols[1] * cos(offset);
      symbols[0] = I;
      symbols[1] = Q;
      I = symbols[2] * cos(offset) - symbols[3] * sin(offset);
      Q = symbols[2] * sin(offset) + symbols[3] * cos(offset);
      symbols[2] = I;
      symbols[3] = Q;
      I = symbols[4] * cos(offset) - symbols[5] * sin(offset);
      Q = symbols[4] * sin(offset) + symbols[5] * cos(offset);
      symbols[4] = I;
      symbols[5] = Q;
      I = symbols[6] * cos(offset) - symbols[7] * sin(offset);
      Q = symbols[6] * sin(offset) + symbols[7] * cos(offset);
      symbols[6] = I;
      symbols[7] = Q;
      I = symbols[8] * cos(offset) - symbols[9] * sin(offset);
      Q = symbols[8] * sin(offset) + symbols[9] * cos(offset);
      symbols[8] = I;
      symbols[9] = Q;
      I = symbols[10] * cos(offset) - symbols[11] * sin(offset);
      Q = symbols[10] * sin(offset) + symbols[11] * cos(offset);
      symbols[10] = I;
      symbols[11] = Q;
      I = symbols[12] * cos(offset) - symbols[13] * sin(offset);
      Q = symbols[12] * sin(offset) + symbols[13] * cos(offset);
      symbols[12] = I;
      symbols[13] = Q;
      I = symbols[14] * cos(offset) - symbols[15] * sin(offset);
      Q = symbols[14] * sin(offset) + symbols[15] * cos(offset);
      symbols[14] = I;
      symbols[15] = Q;

      // Map the angle to a value between 0-7
      thetas[0] = atan2f(symbols[1], symbols[0]);
      thetas[1] = atan2f(symbols[3], symbols[2]);
      thetas[2] = atan2f(symbols[5], symbols[4]);
      thetas[3] = atan2f(symbols[7], symbols[6]);
      thetas[4] = atan2f(symbols[9], symbols[8]);
      thetas[5] = atan2f(symbols[11], symbols[10]);
      thetas[6] = atan2f(symbols[13], symbols[12]);
      thetas[7] = atan2f(symbols[15], symbols[14]);

      // add pi/8 and an additional 2*pi if the angle is negative
		// add pi/8 and an additional 2*pi if the angle is negative
		thetas[0] += PI / 8;
		thetas[1] += PI / 8;
		thetas[2] += PI / 8;
		thetas[3] += PI / 8;
		thetas[4] += PI / 8;
		thetas[5] += PI / 8;
		thetas[6] += PI / 8;
		thetas[7] += PI / 8;

		thetas[0] += (thetas[0] < 0) ? 2 * PI : 0;
		thetas[1] += (thetas[1] < 0) ? 2 * PI : 0;
		thetas[2] += (thetas[2] < 0) ? 2 * PI : 0;
		thetas[3] += (thetas[3] < 0) ? 2 * PI : 0;
		thetas[4] += (thetas[4] < 0) ? 2 * PI : 0;
		thetas[5] += (thetas[5] < 0) ? 2 * PI : 0;
		thetas[6] += (thetas[6] < 0) ? 2 * PI : 0;
		thetas[7] += (thetas[7] < 0) ? 2 * PI : 0;

      // Map the angle to a value between 0-7 with (char)(theta * 8 / (2*PI));
      c[0] = (char)(thetas[0] * 4 / PI);
      c[1] = (char)(thetas[1] * 4 / PI);
      c[2] = (char)(thetas[2] * 4 / PI);
      c[3] = (char)(thetas[3] * 4 / PI);
      c[4] = (char)(thetas[4] * 4 / PI);
      c[5] = (char)(thetas[5] * 4 / PI);
      c[6] = (char)(thetas[6] * 4 / PI);
      c[7] = (char)(thetas[7] * 4 / PI);

      // gray code 
      c[0] = c[0] ^ (c[0] >> 1);
      c[1] = c[1] ^ (c[1] >> 1);
      c[2] = c[2] ^ (c[2] >> 1);
      c[3] = c[3] ^ (c[3] >> 1);
      c[4] = c[4] ^ (c[4] >> 1);
      c[5] = c[5] ^ (c[5] >> 1);
      c[6] = c[6] ^ (c[6] >> 1);
      c[7] = c[7] ^ (c[7] >> 1);

      // write the back to message glued
      messageDecoded[idx_w] = (c[0] << 5) | (c[1] << 2) | (c[2] >> 1);
      messageDecoded[idx_w+1] = (c[2] << 7) | (c[3] << 4) | (c[4] << 1) | (c[5] >> 2);
      messageDecoded[idx_w+2] = (c[5] << 6) | (c[6] << 3) | (c[7]);
   }
}

// glue the 3-bit symbols back together
void gluer(char* finalData, char* stuffedData, unsigned in_len) {
	
   // NOTE: 8 0b00000xxx -> 3 0bxxxxxxxx
	unsigned j = 0;
	for (unsigned i = 0; i < in_len; i += 8) {
      glue(&finalData[j], &stuffedData[i]);
		j += 3;
	}
}