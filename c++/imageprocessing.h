#ifndef IMAGEPROCESSING_H_DEFINED
#define IMAGEPROCESSING_H_DEFINED

#include "utils.h"
#include "png.hpp"

/*
 * Misc
 */

// Draws a smile image to the address specified
void drawASmile(char const* addrOut);
void drawASmile(std::string addrOut);

// Function to create Gaussian kernel on length n
template<class T>
T** gaussiankernel(unsigned const masklength, T const sigma);

/*
 * Cropping
 */

// Crops a given image at the specified coordinates
png::image<png::gray_pixel> crop(png::image<png::gray_pixel> const &img, unsigned const x0, unsigned const x1, unsigned const y0, unsigned const y1);

// Crops the image between two staggered y bounds
//  - Fills the empty space with whitespace
// USAGE:
// Requires two arrays that span the width and specify the top and bottom y values respecively
// (Optional) padding field that will add whitespace above and below the highest and lowest y points
png::image<png::gray_pixel> staggeredYCrop(png::image<png::gray_pixel> const& img, std::vector<unsigned> const& yTopArray, std::vector<unsigned> const& yBotArray, unsigned const yPadding = 0);

// Crops the image between two staggered x bounds
//  - Fills the empty space with whitespace
// USAGE:
// Requires two arrays that span the width and specify the top and bottom x values respecively
// (Optional) padding field that will add whitespace above and below the highest and lowest x points
png::image<png::gray_pixel> staggeredXCrop(png::image<png::gray_pixel> const& img, std::vector<unsigned> const& xTopArray, std::vector<unsigned> const& xBotArray, unsigned const xPadding = 0);

/*
 * Segmentation
 */

// Segment the image into black and white using a fixed threshold
png::image<png::gray_pixel> segmentImageThreshold(png::image<png::gray_pixel> const &imgSrc, unsigned const T);

// Segment the image into black and white using Kittlers method
unsigned determineKittlerThreshold(png::image<png::gray_pixel> const &imgSrc);

/*
 * Histograms
 */

// Create a histogram of the horizontal projection of a grayscale image
std::vector<unsigned> horizontalProjectionHistogram(png::image<png::gray_pixel> const &img, unsigned const thresh = 64, unsigned const binWidth = 1);

// Create a histogram of the horizontal projection of a grayscale image (normalizes the output)
// "depth" specify float vs double
template<typename depth>
std::vector<depth> horizontalProjectionHistogramNorm(png::image<png::gray_pixel> const &img, unsigned const thresh = 64, unsigned const binWidth = 1);

// Create a histogram of the vertical projection of a grayscale image
std::vector<unsigned> verticalProjectionHistogram(png::image<png::gray_pixel> const &img, unsigned const thresh = 64, unsigned const binWidth = 1);

// Create a histogram of the vertical projection of a grayscale image (normalizes the output)
// "depth" specify float vs double
template<typename depth>
std::vector<depth> verticalProjectionHistogramNorm(png::image<png::gray_pixel> const &img, unsigned const thresh = 64, unsigned const binWidth = 1);

// Get a list of midpoints from a given histogram
std::vector<unsigned> getMidPoints(std::vector<unsigned> hist, unsigned minBinCount, unsigned const minGapThresh=0);

/*
 * GRADIENTS
 */

enum gradType{
    SOBEL,
    MEDIAN
};

/*
         [-1][0][1]       [-1][-2][-1]  
    h1 = [-2][0][2]  h2 = [ 0][ 0][ 0]  scaling factor: 1/8
         [-1][0][1]       [ 1][ 2][ 1]

    [A][B][C] 
    [D][E][F] 
    [G][H][I] 

    Returns: 
        Gx, Gy
*/
float computeGradientVector_sobel(png::image<png::gray_pixel> const &image, unsigned const r, unsigned const c, unsigned const width, unsigned const height);
tuple2<float> computeDirectionalGradientVector_sobel(png::image<png::gray_pixel> const &image, unsigned const r, unsigned const c, unsigned const width, unsigned const height);

/*
    𝐼𝑥(𝑟, 𝑐) = median{𝐼(𝑟 − 1, 𝑐),𝐼(𝑟 − 1, 𝑐 + 1),𝐼(𝑟, 𝑐 + 1),𝐼(𝑟 + 1, 𝑐),𝐼(𝑟 + 1, 𝑐 + 1)}
                    − median{𝐼(𝑟 − 1, 𝑐 − 1),𝐼(𝑟 − 1, 𝑐),𝐼(𝑟, 𝑐 − 1),𝐼(𝑟 + 1, 𝑐 − 1),𝐼(𝑟 + 1, 𝑐)}
                    
    𝐼𝑦(𝑟, 𝑐) = median{𝐼(𝑟, 𝑐 − 1),𝐼(𝑟, 𝑐 + 1),𝐼(𝑟 − 1, 𝑐 − 1),𝐼(𝑟 − 1, 𝑐),𝐼(𝑟 − 1, 𝑐 + 1)}
                    − median{𝐼(𝑟, 𝑐 − 1),𝐼(𝑟, 𝑐 + 1),𝐼(𝑟 + 1, 𝑐 − 1),𝐼(𝑟 + 1, 𝑐),𝐼(𝑟 + 1, 𝑐 + 1)}

    [A][B][C]   [A][B]   [B][C]    [A][B][C]          
    [D][E][F]   [D]         [F]    [D]   [F]   [D]   [F]
    [G][H][I]   [G][H]   [H][I]                [G][H][I]
                    x0       x1          y0          y1

    𝐼𝑥(𝑟, 𝑐) = median(x1) - median(x0)
    𝐼𝑦(𝑟, 𝑐) = median(y0) - median(y1)

    Returns: 
        Gx, Gy
*/
float computeGradientVector_median(png::image<png::gray_pixel> const &image, unsigned const r, unsigned const c, unsigned const width, unsigned const height);
tuple2<float> computeDirectionalGradientVector_median(png::image<png::gray_pixel> const &image, unsigned const r, unsigned const c, unsigned const width, unsigned const height);

// Populates a gradient map based on the provided image
std::vector<std::vector<float>> generateGradientMap(png::image<png::gray_pixel> const &image, gradType const gradientType = SOBEL);
// Populates a directional gradient map based on the provided image, read (Gx, Gy)
std::vector<std::vector<tuple2<float>>> generateDirectionalGradientMap(png::image<png::gray_pixel> const &image, gradType const gradientType = SOBEL);

/*
 * Document Cleaning
 */

// Find the yMin and yMax values for the handwritten part of an IAM dataset text document
// NOTE: Cuttoff is made at the line segments, there are 2 on the top and one on the bottom
//       The text will be located between the top 2 and bottom
//           -------------------
//           Computer text
//           -------------------
//           *Handwritten text*
//           *Handwritten text*
//           *Handwritten text*
//           -------------------
tuple4<unsigned> findIAMTextBounds(png::image<png::gray_pixel> const &img);

// Cleans a document from the IAM dataset
// - Crops out everything but the centered text
png::image<png::gray_pixel> cleanIAMDocument(png::image<png::gray_pixel> const &img);

/*
 * Image Filtering
 */

// Create edgemap from the given image
png::image<png::gray_pixel> edgeMapImg(png::image<png::gray_pixel> const &img, gradType const gradientType=SOBEL, float const threshold=60);

// Process that fills in the gaps on our edgemaps
png::image<png::gray_pixel> erodeImg(png::image<png::gray_pixel> const& imgDoc, unsigned const kernelRadius=3);

#endif