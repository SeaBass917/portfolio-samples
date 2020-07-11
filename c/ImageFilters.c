// Sebastian Thiem, Ken Gregg
// ENG276IN
// 23 March 2017
// Implentations of 8 different image filters
// - Negation
// - Grayscale
// - Sepia
// - Binary
// - Solarized
// - Poster
// - Box Blur
// - Gaussian Blur

#define _USE_MATH_DEFINES

#include <stdlib.h>
#include <math.h>
#include "ImageFilters.h"

// Returns the Luminance value of a pixel
uint8_t Luminance(Pixel myPix)
{
	return (uint8_t)floor(myPix.redChannel * 0.2126 + myPix.greenChannel * 0.7152 + myPix.blueChannel * 0.0722);
}

// Negates a pixel
void NegatePixel(Pixel *pFirstPixel)
{
	pFirstPixel->redChannel = 255 - pFirstPixel->redChannel;
	pFirstPixel->greenChannel = 255 - pFirstPixel->greenChannel;
	pFirstPixel->blueChannel = 255 - pFirstPixel->blueChannel;
}

// Converts to negative (note that negating a negative image restores)
bool ConvertImageToNegative(Image *pImage)
{
	// Check that pImage is true
	if (pImage)
	{
		// Loop though the pixels
		for (unsigned i = 0; i < pImage->pixelCount; i++)
		{
			// Check that the pixel is not dead (shouldnt happen but here just incase i goofed
			if (&pImage->pFirstPixel[i])
			{
				// Thres an app for that
				NegatePixel(&pImage->pFirstPixel[i]); 
			}
			else
			{
				// we hit a missing pixel in the array .. somehow
				return false;
			}
		}

		// lo hicimos
		return true;
	}
	else
	{
		// pImage was null
		return false;
	}
}

// Converts to grayscale, returns Grayskull on success
bool ConvertImageToGrayscale(Image *pImage)
{
	// Checks that pImage is non-NULL
	if (pImage)
	{
		// Loops through the pixels
		for (unsigned i = 0; i < pImage->pixelCount; i++)
		{
			// no pixel will be left behind
			if (&pImage->pFirstPixel[i])
			{
				// Calculates luminance and sets each pixel to that value
				uint8_t lum = Luminance(pImage->pFirstPixel[i]);
				pImage->pFirstPixel[i].redChannel = lum;
				pImage->pFirstPixel[i].greenChannel = lum;
				pImage->pFirstPixel[i].blueChannel = lum;
			}
			else
			{
				// A good pixel was lost today, we are returning home
				return false;
			}
		}

		// Mission accomplished
		return true;
	}
	else
	{
		// pImage was NULL
		return false;
	}
}

// Converts image to sepia
bool ConvertImageToSepia(Image *pImage)
{
	// Checks that pImage is non-NULL
	if (pImage)
	{
		// Loops through the pixels
		for (unsigned i = 0; i < pImage->pixelCount; i++)
		{
			// make sure we didn't lose a pixel, and by we I mean you caller
			if (&pImage->pFirstPixel[i])
			{
				// Need a temp pixel so we dont overwrite data
				Pixel pixelTemp;

				pixelTemp.redChannel = (uint8_t)floor(min((pImage->pFirstPixel[i].redChannel * 0.393 + pImage->pFirstPixel[i].greenChannel * 0.769 + pImage->pFirstPixel[i].blueChannel * 0.189), 255));
				pixelTemp.greenChannel = (uint8_t)floor(min((pImage->pFirstPixel[i].redChannel * 0.349 + pImage->pFirstPixel[i].greenChannel * 0.686 + pImage->pFirstPixel[i].blueChannel * 0.168), 255));
				pixelTemp.blueChannel = (uint8_t)floor(min((pImage->pFirstPixel[i].redChannel * 0.272 + pImage->pFirstPixel[i].greenChannel * 0.534 + pImage->pFirstPixel[i].blueChannel * 0.131), 255));

				pImage->pFirstPixel[i] = pixelTemp;
			}
			else
			{
				// Pixel was lost in translation
				return false;
			}
		}

		return true;
	}
	else
	{
		// pImage was NULL
		return false;
	}
}

// Threshhold double [0, 255]
bool ConvertImageToBinary(Image *pImage, double threshold)
{
	// Checks that pImage is non-NULL
	if (pImage && threshold > 0 && threshold < 255)
	{
		// Loops through the pixels
		for (unsigned i = 0; i < pImage->pixelCount; i++)
		{
			// make sure the pixel exists
			if (&pImage->pFirstPixel[i])
			{
				// Calculates luminance
				uint8_t lum = Luminance(pImage->pFirstPixel[i]);

				// if its above set to max: below, min
				if (lum > threshold)
				{
					lum = 255;
				}
				else
				{
					lum = 0;
				}

				// setting it to either max or min
				pImage->pFirstPixel[i].redChannel = lum;
				pImage->pFirstPixel[i].greenChannel = lum;	
				pImage->pFirstPixel[i].blueChannel = lum;
			}
			else
			{
				// 404 pixel not found
				return false;
			}
		}

		// Mission accomplished
		return true;
	}
	else
	{
		// pImage was NULL
		return false;
	}
}

// Threshhold double [0, 255] 
bool ConvertImageToSolarized(Image *pImage, double threshold)
{
	// Checks that pImage is non-NULL
	if (pImage)
	{
		// Caller didnt understand my set notation
		if (threshold < 0 || threshold > 255)
		{
			// now im in charge
			threshold = 150; 
		}

		// Loops through the pixels
		for (unsigned i = 0; i < pImage->pixelCount; i++)
		{
			// Calculates luminance
			uint8_t lum = Luminance(pImage->pFirstPixel[i]);

			if (lum >= threshold)
			{
				NegatePixel(&pImage->pFirstPixel[i]);
			}
		}

		// Mission accomplished
		return true;
	}
	else
	{
		// pImage was NULL
		return false;
	}
}


// Posterize concept in progress
// Posterizes the image Threshold [0, 255] 

bool ConvertImageToPoster(Image *pImage, unsigned level)
{
	// Checks that pImage is non-NULL
	if (pImage)
	{
		unsigned level = 32;

		// Plus one so the array can have 0 and 255
		unsigned *intervals = malloc(((256 / level) + 1)* sizeof(unsigned));

		unsigned intervalsCount = 256 / level;

		intervals[0] = 0;

		for (unsigned i = 1; i <= intervalsCount; i++)
		{
			intervals[i] = level * i - 1;
		}

		// Loops through the pixels
		for (unsigned i = 0; i < pImage->pixelCount; i++)
		{

		}

	// Mission accomplished
	return true;

	}
	else
	{
		// pImage was NULL
		return false;
	}
}



// Creates blurred image
Image *CreateBoxBlurImage(const Image *pImage, unsigned radius)
{
	// Check that pImage is non-NULL
	if (pImage)
	{
		// We need a clone to do this filter
		Image *pImageBlurred = CreateClonedImage(pImage);

		// Make sure clone succeeded
		if (pImageBlurred)
		{
			// call em what they are
			unsigned rows = pImage->infoHeader.bitmapHeight;
			unsigned cols = pImage->infoHeader.bitmapWidth;

			// If the caller has unrealistic expectations then fix that for them
			if (radius < 1)
			{
				radius = 1;
			}
			else if (radius > min(rows, cols) / 2)
			{
				radius = min(rows, cols) / 2;
			}

			// goes through each pixel like a 2dmap
			for (unsigned row = 0; row < rows; row++)
			{
				for (unsigned col = 0; col < cols; col++)
				{
					// Sum of each color channel and the number of pixels being averaged
					double rSum = 0;		
					double gSum = 0;
					double bSum = 0;
					double pixAvgCount = 0;

					// Loop through each neighbour row and column 
					for (int nRow = row - (int)radius; nRow <= row + (int)radius; nRow++)
					{
						for (int nCol = col - (int)radius; nCol <= col + (int)radius; nCol++)
						{
							// Ignore the edges
							if (nRow >= 0 && nRow < rows && nCol >= 0 && nCol < cols)
							{
								// gunna get the avg or each colour nearby
								rSum += (pImage->pFirstPixel + nRow*cols + nCol)->redChannel;
								gSum += (pImage->pFirstPixel + nRow*cols + nCol)->greenChannel;
								bSum += (pImage->pFirstPixel + nRow*cols + nCol)->blueChannel;

								pixAvgCount++;
							}
						}
					}
					// calculate the avg and store it in the new image
					(pImageBlurred->pFirstPixel + row*cols + col)->redChannel = (uint8_t)(rSum / pixAvgCount);
					(pImageBlurred->pFirstPixel + row*cols + col)->greenChannel = (uint8_t)(gSum / pixAvgCount);
					(pImageBlurred->pFirstPixel + row*cols + col)->blueChannel = (uint8_t)(bSum / pixAvgCount);
				}
			}

			// yeahhhh 
			return pImageBlurred;
		}
		else
		{
			// We have failed to clone
			return NULL;
		}
	}
	else
	{
		// pImage was NULL
		return NULL;
	}
}

// gauss's trademarked instagram filter
Image *CreateGaussianBlurImage(const Image *pImage, unsigned radius, double sigma)
{
	// Check that pImage is non-NULL
	if (pImage)
	{
		// We need a clone to do this filter
		Image *pImageGBlurred = CreateClonedImage(pImage);

		// Make sure clone succeeded
		if (pImageGBlurred)
		{
			// call em what they are
			unsigned rows = pImage->infoHeader.bitmapHeight;
			unsigned cols = pImage->infoHeader.bitmapWidth;

			// If the caller has unrealistic expectations then fix that for them
			if (radius < 1)
			{
				radius = 1;
			}
			else if (radius > min(rows, cols) / 2)
			{
				radius = min(rows, cols) / 2;
			}

			// Compute the rows and columns in the Gaussian kernel.
			int kernelDimension = radius * 2 + 1;
			// Compute the total number of elements (doubles) in the Gaussian kernel.
			int elements = kernelDimension * kernelDimension;
			// Allocate the Gaussian kernel array (two-dimensional array of doubles).
			double *pKernel = malloc(elements * sizeof(double));

			if (pKernel)		// Did the allocation succeed?
			{
				// Calculate the values in the Gaussian kernel matrix. Note that we have
				// pulled some of the calculation work up out of the loops. See the project
				// description for details of the formulas used in calculating the values.
				double denominator = 2 * sigma * sigma;
				double factor = 1.0 / (denominator * M_PI);
				double sumTotal = 0.0;
				double *pElement = pKernel;				// points to current kernel element

				// Treat the Gaussian kernel as a two-dimensional array, and compute
				// and store the values into it, element by element. Note that we rely
				// on the fact that C stores two-dimensional arrays in row-major order.

				for (int row = 0; row < kernelDimension; ++row)
				{
					int x = row - radius;
					int xSquared = x * x;

					for (int col = 0; col < kernelDimension; ++col)
					{
						int y = col - radius;
						int ySquared = y * y;

						// Compute the actual value to store into the array.
						*pElement = factor * exp(-(xSquared + ySquared) / (denominator));

						sumTotal += *pElement;  // needed later for normalization
						++pElement;				// move on to next kernel element
					}
				}

				// Now, normalize all values, so that the sum of all values is equal to 1.0.				
				pElement = pKernel;

				for (int rawIndex = 0; rawIndex < elements; ++rawIndex)
				{
					*pElement++ /= sumTotal;
				}

				// Find the center of the kernel, as we'll need this later.
				double *pKernelCenter = pKernel + radius * kernelDimension + radius;

				// goes through each pixel like a 2dmap
				for (unsigned row = 0; row < rows; row++)
				{
					for (unsigned col = 0; col < cols; col++)
					{
					
							// Sum of each color channel and the number of pixels being averaged
							double rSum = 0;
							double gSum = 0;
							double bSum = 0;
							unsigned i = 0;
							//double pixAvgCount = 0;


							// Loop through each neighbour row and column 
							for (unsigned nRow = row - radius; nRow <= row + radius; nRow++)
							{
								for (unsigned nCol = col - radius; nCol <= col + radius; nCol++)
								{
									// Ignore the edges
									if (nRow >= 0 && nRow < rows && nCol >= 0 && nCol < cols)
									{
									rSum += (pImage->pFirstPixel + nRow * cols + nCol)->redChannel * (*(pKernel + i));
									gSum += (pImage->pFirstPixel + nRow * cols + nCol)->greenChannel * (*(pKernel + i));
									bSum += (pImage->pFirstPixel + nRow * cols + nCol)->blueChannel * (*(pKernel + i));
									i++;

									//pixAvgCount += *(pKernel + i);
									}
								}

							}
							// calculate the avg and store it in the new image
							(pImageGBlurred->pFirstPixel + row*cols + col)->redChannel = (uint8_t)rSum;
							(pImageGBlurred->pFirstPixel + row*cols + col)->greenChannel = (uint8_t)gSum;
							(pImageGBlurred->pFirstPixel + row*cols + col)->blueChannel = (uint8_t)bSum;
						}
					}
				}

				free(pKernel);
				return pImageGBlurred;
			}
			else
			{
				// Clone failed
				return NULL;
			}
		}
		else
		{
			// pImage was NULL
			return NULL;
		}
}


