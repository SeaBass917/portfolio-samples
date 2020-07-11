///////////////////////////////////////////////////////////////////////////////////////////////////
// Engineer(s):     Sebastian Thiem, Catherine McIntosh 
//
// Sponsor:         General Dynamics
// 
// Project Name:    GPU Modulation
// File Name:		channel.cu
// Create Date:     24 January 2019
//
// Description:     Channel simulations and modules related to simulating 
//					a communication channel.
//					Notably: Phase rotation and AWGN
//
///////////////////////////////////////////////////////////////////////////////////////////////////
#include "channel.cuh"
#include "util.cuh"
#include "mod.cuh"
#include "demod.cuh"
#include "srrc.cuh"
#include "const.cuh"
#include <omp.h>

// rotate vector 15 degress using linear transformation
void rotate(float* I, float* Q, const float p) {

	 // use temp variable
	 float x_p[2];

	 x_p[0] = *I * cos(p) - *Q * sin(p);
	 x_p[1] = *I * sin(p) + *Q * cos(p);

	 *I = x_p[0];
	 *Q = x_p[1];
}

void simPhaseOffset(float* data, const unsigned len, const float offset) {

	 // Rotate the symbol counter clockwise
	 for (unsigned i = 0; i < len; i += 2) {
		  rotate(&data[i], &data[i + 1], offset);
	 }
}

// Adds noise to a hard coded PACKET_SIZE length float array that is Gaussian and White
void addNoise(float* signal, float stddev, unsigned size) {
	 std::normal_distribution<float> norm(0.0f, stddev);
	 std::mt19937_64 generator(std::random_device{}());

	 // Add Gaussian noise
	 for (unsigned i = 0; i < size; i++) {
		  signal[i] = signal[i] + norm(generator);
	 }
}

// Simulate transmission over a channel of ch SNR
// return BER information
// TODO update this when I finish the one below
unsigned transceiver(const char* dstFileName, const char* srcFileName, unsigned channel, bool addRRC, bool addDelay) {

	 // count of the bit errors
	 unsigned errors = 0;

	 return errors;
}

// Simulate transmission over a noiseless channel
void transceiver(const char* dstFileName, const char* srcFileName, bool addRRC, bool addDelay) {

	 // count of the bit errors
	 unsigned long long fSize = 0;

	 // keeps track of how many packets we've found, 
	 // pulled out so I can use it in analysis
	 unsigned p = 0;

#ifdef PROFILE
	 struct timeval start_demod, stop_demod;
	 struct timeval start_RRC, stop_RRC;
	 struct timeval start_sync, stop_sync;
	 struct timeval start_DS, stop_DS;
	 struct timeval start_decode, stop_decode;
	 unsigned long diffTime_demod = 0;
	 unsigned long diffTime_RRC = 0;
	 unsigned long diffTime_sync = 0;
	 unsigned long diffTime_DS = 0;
	 unsigned long diffTime_decode = 0;
#endif

	 // open message file
	 std::ifstream fMsg(srcFileName, std::ios::in | std::ios::binary);
	 if (fMsg.is_open()) {

		  // Get size of message for preprocessing
		  fSize = getFileSize(srcFileName);

		  // Precalculate the number of packets per file
		  // Rounded down to not include extra bytes that dont fill a packet on their own
		  unsigned packets_in_file = (unsigned)((fSize - 1.0) / PACKET_DATA_SIZE + 1);

		  std::ofstream fDst(dstFileName, std::ios::out | std::ios::binary);
		  if (fDst.is_open()) {

				// leftover 
				unsigned rem = fSize % PACKET_DATA_SIZE;
				unsigned bytes_in_packet = 0;

#ifdef VERBOSE
				std::cout << "Transmitting file: \'" << srcFileName << "\'..." << std::endl << std::endl;

				std::cout << "Simulation Parameters:" << std::endl;
				std::cout << "Message Size: " << fSize << " Bytes" << std::endl;
				std::cout << "Packet Size: " << PACKET_SIZE << " symbols." << std::endl;
				std::cout << "Bytes Per Packet: " << PACKET_DATA_SIZE << " Bytes" << std::endl;
				std::cout << "Packets In File: " << packets_in_file << " packets." << std::endl;
				std::cout << "SNR: INF dB." << std::endl;
				std::cout << "Using RRC filter? " << ((addRRC) ? "true." : "false.") << std::endl;
				std::cout << "Simulating delay between packets? " << ((addDelay) ? "true." : "false.") << std::endl << std::endl;
#endif

				// ---------------------------------------- Host Memory structures ----------------------------------------
				char* buffer;           // Buffer read from the file, large enough to fill a packet [PACKET_DATA_SIZE]
				char* cutBuffer;        // array to hold the binary symbols [DS_PACKET_SIZE]
				float* symbolPacket;    // Array to hold the complex symbols (upsampled) [PACKET_SIZE][2]
				float* full_header;     // full_header with preamble and filesize [FULL_HEADER_SYMBOL_COUNT][2]
				float* symbolsSent;     // full IQ message with packet full_header [PACKET_SIZE + FULL_HEADER_SYMBOL_COUNT][2]
				float* symbolsRec;      // this pointer will hold the array of data "received'
				float* header;
				char* header_c;

				float* downsampledMessage; // The downsample IQ recieved at the reciever
				char* messageDecoded;      // Relevant bits are glued back together in coalesced memory 0bxxxxxxxx

				// ---------------------------------------- Device Memory structures ----------------------------------------
				float* d_symbolsRec;
				float* d_downsampledMessage;
				char* d_messageDecoded;

				// ---------------------------------------- Allocating GPU space ------------------------------------
				// This GPU is big I should be able to allocate larger spaces up here, once, rather than all the time

				cudaMalloc((void**)&d_symbolsRec, 2 * RECIEVER_WIDTH * sizeof(float));
				cudaMalloc((void**)&d_downsampledMessage, 2 * DS_PACKET_SIZE * sizeof(float));
				cudaMalloc((void**)&d_messageDecoded, PACKET_DATA_SIZE * sizeof(char));

				dim3 dimBlock_RRC(1024);
				dim3 dimGrid_RRC((unsigned)(RECIEVER_WIDTH - (MASK_WIDTH - 1) - 1.0) / 1024 + 1);

				dim3 blockDim_DS(BLOCK_WIDTH);
				dim3 gridDim_DS((unsigned)((DS_PACKET_SIZE - 1) / BLOCK_WIDTH + 1));

				dim3 blockDim_demod(BLOCK_WIDTH);
				dim3 gridDim_demod((unsigned)((PACKET_DATA_SIZE / 3 - 1) / BLOCK_WIDTH + 1));

				// Loop through each packet in the file
				// modulate the packet, simulate a channel, demodulate the packet, then monitor the BER/throughput
				for (p = 0; p < packets_in_file; p++) {

					 // ---------------------------------------- Modulation ----------------------------------------

					 bytes_in_packet = (p < packets_in_file - 1) ? PACKET_DATA_SIZE : rem;

					 // allocate memory
					 cutBuffer = (char*)calloc(DS_PACKET_SIZE, sizeof(char));
					 buffer = (char*)calloc(PACKET_DATA_SIZE, sizeof(char));
					 symbolPacket = (float*)calloc(PACKET_SIZE * 2, sizeof(float));
					 full_header = (float*)calloc(FULL_HEADER_SYMBOL_COUNT * 2, sizeof(float));
					 symbolsSent = (float*)malloc(FULL_HEADER_SIZE + PACKET_IQ_SIZE);

					 // read in a buffer
					 fMsg.read(buffer, bytes_in_packet);

					 // Cut up the buffer to symbols
					 cutter(cutBuffer, buffer, PACKET_DATA_SIZE);

					 // map binary data to, gray coded, complex symbols
					 mapper(symbolPacket, cutBuffer, PACKET_SIZE * 2);

					 // get full_header, file size is full packet
					 getHeader(full_header, bytes_in_packet);

					 // basically glue the full_header and packet into one memory block
					 memcpy(symbolsSent, full_header, FULL_HEADER_SIZE);
					 memcpy(&symbolsSent[FULL_HEADER_SYMBOL_COUNT * 2], symbolPacket, PACKET_IQ_SIZE);

					 // remove the old memory blocks
					 free(full_header);
					 free(symbolPacket);
					 free(cutBuffer);

					 // pad zeros around the preserve filter information on edges
					 // pad more than we need to also simulate delay
					 symbolsRec = (float*)calloc(2 * RECIEVER_WIDTH, sizeof(float));
					 memcpy(&symbolsRec[2 * DELAY_US + MASK_WIDTH_0 - 1], symbolsSent, FULL_HEADER_SIZE + PACKET_IQ_SIZE);

					 if (addRRC) {
						  /*cudaMemcpy(d_symbolsRec, symbolsRec, 2 * RECIEVER_WIDTH * sizeof(float), cudaMemcpyHostToDevice);
						  convolution_kernel << <dimGrid_RRC, dimBlock_RRC >> > (d_symbolsRec, RECIEVER_WIDTH);
						  cudaMemcpy(symbolsRec, d_symbolsRec, 2 * RECIEVER_WIDTH * sizeof(float), cudaMemcpyDeviceToHost);*/
						  convolution(symbolsRec, RECIEVER_WIDTH);
					 }

					 // ---------------------------------------- Transmit ----------------------------------------
					 unsigned packet_size = 0;  // bytes of data in the packet; (embedded in the full_header)

					 // Send the packet to the reciever and wait for a responce
					 bool confirmation = false;
					 unsigned timeout = 24;
					 unsigned t = 0;
					 while (!confirmation && t < timeout) {

						  // ---------------------------------------- Channel ----------------------------------------
						  // float symbolsSent[PACKET_SIZE + FULL_HEADER_SYMBOL_COUNT][2]

						  // simulate offset (only once though)
						  // TODO come back to this and have it POfsett += new Poffset 
						  // so we know what to expect but also it changes each send
						  if (t == 0) {
								simPhaseOffset(symbolsRec, 2 * RECIEVER_WIDTH, POFFSET);
						  }
#ifdef PROFILE 
						  gettimeofday(&start_demod, NULL);
#endif
						  // ---------------------------------------- Filter ----------------------------------------
						  // float symbolsRec[symbolsRec_count]

						  // FIlter the recieved information
						  if (addRRC) {
								if (t == 0) {
#ifdef PROFILE 
									 gettimeofday(&start_RRC, NULL);
#endif
#ifdef NVIDIA
									 /*cudaMemcpy(d_symbolsRec, symbolsRec, 2 * RECIEVER_WIDTH * sizeof(float), cudaMemcpyHostToDevice);
									 convolution_kernel << <dimGrid_RRC, dimBlock_RRC >> > (d_symbolsRec, RECIEVER_WIDTH);
									 cudaMemcpy(symbolsRec, d_symbolsRec, 2 * RECIEVER_WIDTH * sizeof(float), cudaMemcpyDeviceToHost);
									 writeSignalToCSV(symbolsRec, "rx_filter", 2 * RECIEVER_WIDTH);*/
									 convolution(symbolsRec, RECIEVER_WIDTH);
#else
									 convolution(symbolsRec, RECIEVER_WIDTH);
#endif
#ifdef PROFILE  
									 gettimeofday(&stop_RRC, NULL);
									 diffTime_RRC += diff_time_usec(start_RRC, stop_RRC);
#endif 
								}
						  }

						  // ---------------------------------------- Syncronize ----------------------------------------
#ifdef PROFILE 
						  gettimeofday(&start_sync, NULL);
#endif
						  // Find the preamble
						  bool preamble_found = false;
						  unsigned sample = 0;
						  unsigned sample_max = 2 * (RECIEVER_WIDTH - PACKET_SIZE);   // if we pass this while searching we've gone too far
						  unsigned i = 0;
						  float d = 0;
						  float d_compliment = 0;
						  unsigned n = 0;
						  float phaseOffset = 0.0f;
						  unsigned tolerance = PREAMBLE_TOLERANCE; // how many errors in the preamble will be tolerate
						  while (!preamble_found && sample < sample_max) {

								d = phaseDiff(
									 symbolsRec[sample], symbolsRec[sample + 1],
									 symbolsRec[sample + DSRATE * 2], symbolsRec[sample + DSRATE * 2 + 1]
								);

								d_compliment = 2 * PI - d;

								// if the difference in angle is ~72
								// start looking for the preamble
								// 72 +/- 10 degrees
								if ((preamble_offsets[n] - P_THRESH_0) < d && d < (preamble_offsets[n] + P_THRESH_0)
									 || ((preamble_offsets[n] - P_THRESH_0) < d_compliment && d_compliment < (preamble_offsets[n] + P_THRESH_0))) {

									 i = DSRATE * 2;
									 n = 1;

									 // begin searching for the preamble
									 while (n && i < 2 * PREAMBLE_SYMBOL_COUNT_UPSAMPLED) {

										  d = phaseDiff(symbolsRec[sample + i + DSRATE * 2], symbolsRec[sample + i + DSRATE * 2 + 1],
												symbolsRec[sample + i], symbolsRec[sample + i + 1]);

										  d_compliment = 2 * PI - d;

										  // Check angle of next sample accept tolerance on bad symbols 
										  // tolerate only angles up to +/- 45 deg as anymore would be improbable at 0dB
										  if ((preamble_offsets[n] - P_THRESH_0) < d && d < (preamble_offsets[n] + P_THRESH_0)
												|| ((preamble_offsets[n] - P_THRESH_0) < d_compliment && d_compliment < (preamble_offsets[n] + P_THRESH_0))) {
												n++;
										  }
										  else if (tolerance > 0 && ((preamble_offsets[n] - P_THRESH_1) < d && d < (preamble_offsets[n] + P_THRESH_1)
												|| ((preamble_offsets[n] - P_THRESH_1) < d_compliment && d_compliment < (preamble_offsets[n] + P_THRESH_1)))) {
												n++;
												tolerance--;
										  }
										  else {
												tolerance = PREAMBLE_TOLERANCE;
												n = 0;
										  }

										  // PREAMBLE_SYMBOL_COUNT - 1 in a row is the preamble
										  if (n >= PREAMBLE_SYMBOL_COUNT - 2) {
												preamble_found = true;
										  }

										  //move 2 floats over, upsampled
										  i += DSRATE * 2;
									 }
								}

								// if we found the preamble, this is the first sample of the header
								// otherwise just move to the next sample
								if (preamble_found == true) {
									 sample += i;
								}
								else {
									 sample += 2;
								}
						  }

						  // check that there is a full packet left after finding the preamble (for memory bounds)
						  if (preamble_found == true) {
								float offset_found = 0;
								// find phase offset as the average of the expected preamble values
								unsigned preamble_start = sample - 2 * PREAMBLE_SYMBOL_COUNT_UPSAMPLED;
								for (unsigned pa = 0; pa < PREAMBLE_SYMBOL_COUNT; pa++) {
									 offset_found = phaseDiff(preamble[pa][0], preamble[pa][1], symbolsRec[preamble_start + 2 * DSRATE * pa], symbolsRec[preamble_start + 2 * DSRATE * pa + 1]);
									 // NEEDS FIX: sometimes the phase difference is 2*PI due to atan2 being shitty
									 phaseOffset += (offset_found) >= 2*PI - 0.02 ? offset_found - 2*PI : offset_found;
								}
								phaseOffset /= PREAMBLE_SYMBOL_COUNT;

#ifdef VERBOSE
								if (preamble_found && abs(POFFSET - phaseOffset) > 0.087f) {
									 std::cout << "\tERROR! Phase offset was: " << POFFSET << " Reciever found: " << phaseOffset << std::endl;
								}
#endif
								// Extract full_header info (like a mini demod)
								header = (float*)malloc(HEADER_MOD_COUNT * 2 * sizeof(float));
								header_c = (char*)malloc(HEADER_PADDED_SIZE * 2 * sizeof(float));
								downsample(header, &symbolsRec[sample], HEADER_MOD_COUNT);
								simPhaseOffset(header, HEADER_MOD_COUNT, -phaseOffset);
								decoder(header_c, header, HEADER_PADDED_SIZE);
								reinterpret_cast<char*>(&packet_size)[1] = header_c[0];
								reinterpret_cast<char*>(&packet_size)[0] = header_c[1];
#ifdef PROFILE  
								gettimeofday(&stop_sync, NULL);
								diffTime_sync += diff_time_usec(start_sync, stop_sync);
#endif 
								// check that packet size interpereted is reasonable
								if (packet_size <= PACKET_DATA_SIZE) {

									 // ------------------------------------------------ Downsample ------------------------------------------------

									 // allocate space for downsampled message / calloc cuz we dont wanna move bad memory
									 downsampledMessage = (float*)malloc(2 * DS_PACKET_SIZE * sizeof(float));
#ifdef PROFILE 
									 gettimeofday(&start_DS, NULL);
#endif
									 unsigned idx = 0;
									 sample += HEADER_MOD_COUNT_UPSAMPLED * 2;
									 for (unsigned id = 0; id < 2 * DS_PACKET_SIZE; id += 2) {

										  // 2 floats per symbol DSRATE symbols per sample
										  idx = DSRATE * id;

										  // write it to the downsampled array
										  downsampledMessage[id] = symbolsRec[sample + idx];
										  downsampledMessage[id + 1] = symbolsRec[sample + idx + 1];
									 }
#ifdef PROFILE  
									 gettimeofday(&stop_DS, NULL);
									 diffTime_DS += diff_time_usec(start_DS, stop_DS);
#endif 
									 // free upsampled 
									 free(symbolsRec);

									 // ---------------------------------------- Demodulate ----------------------------------------

									 // allocate
									 messageDecoded = (char*)malloc(PACKET_DATA_SIZE * sizeof(char));
#ifdef PROFILE 
									 gettimeofday(&start_decode, NULL);
#endif
#ifdef NVIDIA
									 // we're also fixing the phase offset inside this kernel
									 cudaMemcpy(d_downsampledMessage, downsampledMessage, DS_PACKET_SIZE * 2 * sizeof(float), cudaMemcpyHostToDevice);
									 decoder_kernel << <blockDim_demod, gridDim_demod >> > (d_messageDecoded, d_downsampledMessage, PACKET_DATA_SIZE / 3, -phaseOffset);
									 cudaMemcpy(messageDecoded, d_messageDecoded, PACKET_DATA_SIZE * sizeof(char), cudaMemcpyDeviceToHost);
#else
									 decoder(messageDecoded, downsampledMessage, PACKET_DATA_SIZE / 3, -phaseOffset);
#endif
#ifdef PROFILE  
									 gettimeofday(&stop_decode, NULL);
									 diffTime_decode += diff_time_usec(start_decode, stop_decode);
#endif 
									 // free structures
									 free(downsampledMessage);

									 // write the decoded message to the destination file
									 fDst.write(reinterpret_cast<char*>(messageDecoded), packet_size);

									 free(messageDecoded);
#ifdef PROFILE  
									 gettimeofday(&stop_demod, NULL);
									 diffTime_demod += diff_time_usec(start_demod, stop_demod);
#endif 
									 confirmation = true;
								}
								else {
#ifdef VERBOSE
									 std::cout << "<BaHe>";
									 //std::cout << "\tERROR Interpereted invalid packet size <" << packet_size << ">." << std::endl;
#endif
								}
						  }
						  else {
#ifdef VERBOSE
								std::cout << "<NoPre>";
#endif
						  }
						  t++;
					 }

					 // free buffer for packet
					 free(symbolsSent);
					 free(buffer);

					 if (!confirmation) {
#ifdef VERBOSE
						  std::cout << "\n\tERROR Timeout." << std::endl;
						  std::cout << "\tCommunication terminated on packet: " << p << '/' << packets_in_file << '.' << std::endl;
#endif
					 }
				}

				// free cuda stuff 
				cudaFree(d_downsampledMessage);
				cudaFree(d_symbolsRec);
				cudaFree(d_messageDecoded);
		  }

		  // Close the message file
		  fMsg.close();

#ifdef PROFILE
		  std::cout << "Avg Execution Times:" << std::endl;
		  std::cout << "RRC: " << float(diffTime_RRC) / float(p) << " us." << std::endl;
		  std::cout << "Syncronization: " << float(diffTime_sync) / float(p) << " us." << std::endl;
		  std::cout << "Downsample: " << float(diffTime_DS) / float(p) << " us." << std::endl;
		  std::cout << "Decode: " << float(diffTime_decode) / float(p) << " us." << std::endl << std::endl;
		  std::cout << "Demod: " << float(diffTime_demod) / float(p) << " us." << std::endl << std::endl;

		  std::cout << "Demod took: " << diffTime_demod << " us." << std::endl;
		  std::cout << "Effective throughput of demod system: " << 8 * float(fSize) / float(diffTime_demod) << " Mb/s." << std::endl;
#endif // PROFILE

	 }
	 else {
		  std::cout << "\tERROR! Failed to open file \"" << srcFileName << "\"." << std::endl;
		  std::cout << "Exiting Program..." << std::endl;
	 }
}

void channel_sweep(std::string outDir, std::string inFile) {
	 // used for outputting recieved messages
	 std::string outFile;

	 // error counter used in generating ber
	 // we need a minimum of 100 errors to consider a ber sample valid
	 unsigned long long messageSize = getFileSize(inFile.c_str());
	 unsigned long long errs = 0;
	 unsigned runs = 0;
	 unsigned long long bits_processed = 0;
	 //unsigned timeout = 50;     // I dont have all day
	 unsigned timeout = 5000;     // I do
	 float ber = 0;

	 // Loop through decreasing SNR (increasing noise)
	 // calculate errors
	 for (int ch = SNR_MAX; ch >= SNR_MIN; ch--) {
		  bits_processed = 0;
		  errs = 0;
		  runs = 0;

		  // set the dst dir for the current channel count
		  outFile = outDir + "message_recieved_" + std::to_string(ch) + inFile.substr(inFile.find_last_of("."));	// Add file format getter so we can save it as that type

		  while (errs < 100 && runs < timeout) {

				// simulate transmission returning the total number of errors in the transmission
				errs += transceiver(outFile.c_str(), inFile.c_str(), ch, true, true);
				runs++;
				bits_processed += getFileSize(outFile.c_str()) * 8;
		  }

		  ber = errs / ((float)bits_processed);

		  if (errs < 100) {
#ifdef VERBOSE
				std::cout << "\tERROR! Timeout on BER calc. Youare gunna have to run me longer than " << timeout << " trials... sorry." << std::endl;
#endif
				ber = -1;
		  }


		  writeBERToCSV(ch, ber, outDir);
	 }
}
