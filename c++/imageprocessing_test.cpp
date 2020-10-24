#include "../src/imageprocessing.h"
#include "../src/utils.h"

#include <iostream>
#include <string>

// TODO: Automate test a little bit
void edgeDetection_test(std::string testDir, std::string testImg){

    // Address for test image edgemaps
    std::string testPathNoExt = testDir + getPathNoExtension(testImg);
    std::string pathEdgeSobel = testPathNoExt+"_edge-sobel.png";
    std::string pathEdgeMedian = testPathNoExt+"_edge-median.png";

    png::image<png::gray_pixel> img(testImg);

    // Sobel Edge Map
    png::image<png::gray_pixel> edgeMapImage = edgeMapImg(img, SOBEL, 30);
    edgeMapImage.write(pathEdgeSobel);

    // Median Edge Map
    edgeMapImage = edgeMapImg(img, MEDIAN, 40);
    edgeMapImage.write(pathEdgeMedian);
}

void cleanIAMDocument_test(std::string testDir){

    // Test image
    std::string testImg = "../data/IAM/documents/a02-000.png";

    if(fs::exists(testImg)){

        png::image<png::gray_pixel> imgDoc(testImg);

        png::image<png::gray_pixel> imgDocClean = cleanIAMDocument(imgDoc);

        std::string fileOut = testDir+"document_clean.png";
        imgDocClean.write(fileOut);
    }
    else{
        std::cerr << "\tERROR! preProcessing_test() cannot find \""<<testImg<<"\" needed for test." << std::endl;
    }
}

void erosion_test(std::string testDir, std::string testDocumentPath){
    if(fs::exists(testDocumentPath)){
        std::string testDocumentPathNoExt = testDir + getPathNoExtension(testDocumentPath);

        png::image<png::gray_pixel> imgDoc(testDocumentPath);
        
        png::image<png::gray_pixel> imgEdge = edgeMapImg(imgDoc, SOBEL, 60);
        png::image<png::gray_pixel> imgEroded = erodeImg(imgEdge, 3);

        imgEdge.write(testDocumentPathNoExt+"_edge.png");
        imgEroded.write(testDocumentPathNoExt+"_eroded.png");
    }
    else{
        std::cerr << "\tERROR! preProcessing_test() cannot find \""<<testDocumentPath<<"\" needed for test." << std::endl;
    }
}

int main(int argc, char const *argv[]){

    // Create a directory for doing tests
    std::string testDir = "imageprocessing_test/";
    if(!fs::exists(testDir)) fs::create_directory(testDir);

    // Draw a smile
    std::string pathSmile = testDir+"smile.png";
    drawASmile(pathSmile);

    std::string testDir0 = testDir+"edgeDetection_test/";
    if(!fs::exists(testDir0)) fs::create_directory(testDir0);
    edgeDetection_test(testDir0, "document.png");

    std::string testDir1 = testDir+"cleanIAMDocument_test/";
    if(!fs::exists(testDir1)) fs::create_directory(testDir1);
    cleanIAMDocument_test(testDir1);

    std::string testDir2 = testDir+"erosion_test/";
    if(!fs::exists(testDir2)) fs::create_directory(testDir2);
    erosion_test(testDir2, "document_cleaned.png");

    return 0;
}