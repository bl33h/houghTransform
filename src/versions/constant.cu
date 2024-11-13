/*---------------------------------------------------------------------------
Copyright (C), 2024-2025, bl33h, Mendezg1, MelissaPerez09
@author Sara Echeverria, Ricardo Mendez, Melissa Perez
FileName: constant.cu
@version: I
Creation: 05/11/2024
Last modification: 12/11/2024
------------------------------------------------------------------------------*/
#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>
#include <stdlib.h>
#include <string.h>
#include "../pgm/pgm.h"
#include <stdio.h>
#include <math.h>
#include <cuda.h>

const int rBins = 100;
const int degreeInc = 2;
const int degreeBins = (180 / degreeInc);
const float radInc = ((degreeInc * M_PI) / 180);

// CPU_HoughTran function, which calculates the Hough transform sequentially.
void CPU_HoughTran(unsigned char *pic, int w, int h, int **acc) {

  // calculation of the maximum r to be used and memory allocation.
  float rMax = sqrt(1.0 * w * w + 1.0 * h * h) / 2;
  *acc = new int[rBins * degreeBins];
  memset(*acc, 0, (sizeof(int) * rBins * degreeBins));

  // calculation of the image center to use as the origin.
  int xCent = (w / 2);
  int yCent = (h / 2);
  float rScale = ((2 * rMax) / rBins);

  // iteration over the image pixels.
  for (int i = 0; i < w; i++) {
    for (int j = 0; j < h; j++) {

      // calculation of the index to use in the iteration.
      int idx = ((j * w) + i);

      // check that the image pixel is not black to calculate the transform.
      if (pic[idx] > 0) {

        // x and y coordinates.
        int xCoord = (i - xCent);
        int yCoord = (yCent - j);

        // initial theta to use as a test.
        float theta = 0;

        // iteration over the angle range to use.
        for (int tIdx = 0; tIdx < degreeBins; tIdx++) {

          // calculation of the r value for the iteration.
          float r = ((xCoord * cos(theta)) + (yCoord * sin(theta)));

          // calculation of the index to test.
          int rIdx = ((r + rMax) / rScale);

          // increment the used index.
          (*acc)[rIdx * degreeBins + tIdx]++;

          // increase theta by the configured increment.
          theta += radInc;
        }
      }
    }
  }
}

// declaration of constant memory variables.
__constant__ float d_Cos[degreeBins];
__constant__ float d_Sin[degreeBins];

// kernel of the program used to parallelize the Hough transform calculation process.
__global__ void GPU_HoughTran(unsigned char *pic, int w, int h, int *acc, float rMax, float rScale) {

  // calculation and verification that the ID is valid.
  int gloID = threadIdx.x + blockIdx.x * blockDim.x;
  if (gloID > (w * h)) return;

  // calculation of the image center.
  int xCenter = (w / 2);
  int yCenter = (h / 2);

  // coordinate or pixel to be used in this kernel.
  int xCoord = ((gloID % w) - xCenter);
  int yCoord = (yCenter - (gloID / w));

  // check that the iterated pixel is not black.
  if (pic[gloID] > 0) {

    // for loop iterating over the angles to be tested by the kernel.
    for (int tIdx = 0; tIdx < degreeBins; tIdx++) {

      // calculation of the r to test and adding the result.
      float r = ((xCoord * d_Cos[tIdx]) + (yCoord * d_Sin[tIdx]));
      int rIdx = ((r + rMax) / rScale);

      // barrier to synchronize threads within the block.
      __syncthreads();

      // updating the accumulator.
      atomicAdd(acc + (rIdx * degreeBins + tIdx), 1);
    }
  }
}

// function to draw the detected lines on the original image and save it
void drawAndSaveLines(const char *outputFileName, unsigned char *originalImage, int w, int h, int *h_hough, float rScale, float rMax, int threshold) {

  // create an instance of the image to be generated.
  cv::Mat img(h, w, CV_8UC1, originalImage);
  cv::Mat imgColor;
  cvtColor(img, imgColor, cv::COLOR_GRAY2BGR);

  // calculation of the image center.
  int xCenter = (w / 2);
  int yCenter = (h / 2);

  // vector to store lines along with their respective weights.
  std::vector<std::pair<cv::Vec2f, int>>linesWithWeights;

  // iteration to fill the vector with the found lines.
  for (int rIdx = 0; rIdx < rBins; rIdx++) {
    for (int tIdx = 0; tIdx < degreeBins; tIdx++) {

      // weight to place.
      int weight = h_hough[((rIdx * degreeBins) + tIdx)];

      // push the obtained line.
      if (weight > 0) {
        float localReValue = ((rIdx * rScale) - rMax);
        float theta = (tIdx * radInc);
        linesWithWeights.push_back(std::make_pair(cv::Vec2f(theta, localReValue), weight));
      }
    }
  }

  // process to sort the lines by weight in descending order.
  std::sort(
    linesWithWeights.begin(),
    linesWithWeights.end(),
    [](const std::pair<cv::Vec2f, int> &a, const std::pair<cv::Vec2f, int> &b) {
      return a.second > b.second;
    }
  );

  // loop to draw the first obtained lines.
  for (int i = 0; i < std::min(threshold, static_cast<int>(linesWithWeights.size())); i++) {

    // necessary values to obtain the line.
    cv::Vec2f lineParams = linesWithWeights[i].first;
    float theta = lineParams[0];
    float r = lineParams[1];

    // cosine and sine of the iterated angle.
    double cosTheta = cos(theta);
    double sinTheta = sin(theta);

    // values in X and Y, i.e., the found point.
    double xValue = (xCenter - (r * cosTheta));
    double yValue = (yCenter - (r * sinTheta));
    double alpha = 1000;

    // create the line with OpenCV.
    cv::line(
      imgColor,
      cv::Point(cvRound(xValue + (alpha * (-sinTheta))),
      cvRound(yValue + (alpha * cosTheta))),
      cv::Point(cvRound(xValue - (alpha * (-sinTheta))),
      cvRound(yValue - (alpha * cosTheta))),
      cv::Scalar(255, 0, 0),
      1,
      cv::LINE_AA
    );
  }

  // save the image with the detected lines.
  cv::imwrite(outputFileName, imgColor);
}

// main function responsible for executing the program.
int main(int argc, char **argv) {

  // error checking in case an image is not passed.
  if (argc < 3) {
    printf("!WARNING [usage: ./hough <pgm-image> <threshold>]\n");
    return EXIT_FAILURE;
  }

  // load the image passed as a console argument.
  PGMImage inImg(argv[1]);

  // load the threshold to be used.
  int threshold = strtol(argv[2], NULL, 10);

  // get the width and height of the image.
  int *cpuht;
  int w = inImg.x_dim;
  int h = inImg.y_dim;

  // call the sequential version of the Hough transform calculation.
  CPU_HoughTran(inImg.pixels, w, h, &cpuht);

  // get pcCos, pcSin and the initial radians.
  float *pcCos = (float*) malloc(sizeof(float) * degreeBins);
  float *pcSin = (float*) malloc(sizeof(float) * degreeBins);
  float rad = 0;

  // loop to get the cosine and sine of the given radians up to the limit.
  for (int i = 0; i < degreeBins; i++) {
    pcCos[i] = cos(rad);
    pcSin[i] = sin(rad);
    rad += radInc;
  }

  // get the maximum r values for the CUDA version of the transform.
  float rMax = sqrt(1.0 * w * w + 1.0 * h * h) / 2;
  float rScale = ((2 * rMax) / rBins);

  // copy memory of the arrays to be used in constant memory.
  cudaMemcpyToSymbol(d_Cos, pcCos, sizeof(float) * degreeBins);
  cudaMemcpyToSymbol(d_Sin, pcSin, sizeof(float) * degreeBins);

  // instance of values to pass to the parallel version.
  unsigned char *d_in, *h_in;
  int *d_hough, *h_hough;

  // get the image pixels.
  h_in = inImg.pixels;

  // allocate memory for the process.
  h_hough = (int*) malloc(degreeBins * rBins * sizeof(int));

  // allocate memory on the GPU for the variables to use.
  cudaMalloc((void **) &d_in, (sizeof(unsigned char) * w * h));
  cudaMalloc((void **) &d_hough, (sizeof(int) * degreeBins * rBins));
  cudaMemcpy(d_in, h_in, (sizeof(unsigned char) * w * h), cudaMemcpyHostToDevice);
  cudaMemset(d_hough, 0, (sizeof(int) * degreeBins * rBins));

  // calculate the number of blocks to use.
  int blockNum = ceil((w * h) / 256);

  // instance of events and elapsed time.
  cudaEvent_t start, stop;
  float elapsedTime;

  // create CUDA events.
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  // mark the start event.
  cudaEventRecord(start, 0);

  // call the program kernel.
  GPU_HoughTran<<<blockNum, 256>>>(d_in, w, h, d_hough, rMax, rScale);

  // mark the finish event.
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);

  // calculate elapsed time.
  cudaEventElapsedTime(&elapsedTime, start, stop);

  // copy back the results calculated by the GPU.
  cudaMemcpy(h_hough, d_hough, (sizeof(int) * degreeBins * rBins), cudaMemcpyDeviceToHost);

  // print values that differ between CPU and GPU.
  for (int i = 0; i < (degreeBins * rBins); i++) {
    if (cpuht[i] != h_hough[i]) {
      printf("➢ calculation mismatch at: %i %i %i\n", i, cpuht[i], h_hough[i]);
    }
  }

  // print a completion message.
  printf("✓ done\n");

  // print elapsed time.
  printf("-> kernel execution time: %f ms\n", elapsedTime);

  // process to draw the image with the found lines.
  drawAndSaveLines("results/constantOutput.jpg", inImg.pixels, w, h, h_hough, rScale, rMax, threshold);

  // free memory on the GPU.
  cudaFree(d_in);
  cudaFree(d_hough);
  cudaFree(d_Cos);
  cudaFree(d_Sin);

  // free the space used for the process.
  free(h_hough);

  // free memory on the CPU.
  delete[] cpuht;

  // return success.
  return 0;
}