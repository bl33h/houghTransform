/*---------------------------------------------------------------------------
Copyright (C), 2024-2025, bl33h, Mendezg1, MelissaPerez09
@author Sara Echeverria, Ricardo Mendez, Melissa Perez
FileName: shared.cu
@version: I
Creation: 05/11/2024
Last modification: 12/11/2024
------------------------------------------------------------------------------*/
#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>
#include "../pgm/pgm.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <cuda.h>

const int degreeInc = 2;
const int degreeBins = (180 / degreeInc);
const int rBins = 100;
const float radInc = ((degreeInc * M_PI) / 180);

void CPU_HoughTran(unsigned char *pic, int w, int h, int **acc) {

  float rMax = sqrt(1.0 * w * w + 1.0 * h * h) / 2;
  *acc = new int[rBins * degreeBins];
  memset(*acc, 0, (sizeof(int) * rBins * degreeBins));

  int xCent = (w / 2);
  int yCent = (h / 2);
  float rScale = ((2 * rMax) / rBins);

  for (int i = 0; i < w; i++) {
    for (int j = 0; j < h; j++) {

      int idx = ((j * w) + i);

      if (pic[idx] > 0) {

        int xCoord = (i - xCent);
        int yCoord = (yCent - j);

        float theta = 0;

        for (int tIdx = 0; tIdx < degreeBins; tIdx++) {

          float r = ((xCoord * cos(theta)) + (yCoord * sin(theta)));
          int rIdx = ((r + rMax) / rScale);
          (*acc)[rIdx * degreeBins + tIdx]++;
          theta += radInc;
        }
      }
    }
  }
}

__constant__ float d_Cos[degreeBins];
__constant__ float d_Sin[degreeBins];

__global__ void GPU_HoughTran(unsigned char *pic, int w, int h, int *acc, float rMax, float rScale) {

  int localID = threadIdx.x;
  int gloID = localID + blockIdx.x * blockDim.x;

  __shared__ int localAcc[(degreeBins * rBins)];

  for (int i = localID; i < (degreeBins * rBins); i += blockDim.x) {
    localAcc[i] = 0;
  }

  __syncthreads();

  if (gloID < (w * h)) {

    int xCenter = (w / 2);
    int yCenter = (h / 2);

    int xCoord = ((gloID % w) - xCenter);
    int yCoord = (yCenter - (gloID / w));

    if (pic[gloID] > 0) {

      for (int tIdx = 0; tIdx < degreeBins; tIdx++) {

        float theta = tIdx * radInc;
        float r = ((xCoord * d_Cos[tIdx]) + (yCoord * d_Sin[tIdx]));
        int rIdx = (int)((r + rMax) / rScale);
        if ((rIdx >= 0) && (rIdx < rBins)) {
          atomicAdd(acc + (rIdx * degreeBins + tIdx), 1);
        }
      }
    }
  }

  __syncthreads();

  for (int i = localID; i < (degreeBins * rBins); i += blockDim.x) {
    atomicAdd(&acc[i], localAcc[i]);
  }
}

void drawAndSaveLines(const char *outputFileName, unsigned char *originalImage, int w, int h, int *h_hough, float rScale, float rMax, int threshold) {

  cv::Mat img(h, w, CV_8UC1, originalImage);
  cv::Mat imgColor;
  cvtColor(img, imgColor, cv::COLOR_GRAY2BGR);

  int xCenter = (w / 2);
  int yCenter = (h / 2);

  std::vector<std::pair<cv::Vec2f, int>>linesWithWeights;

  for (int rIdx = 0; rIdx < rBins; rIdx++) {
    for (int tIdx = 0; tIdx < degreeBins; tIdx++) {

      int weight = h_hough[((rIdx * degreeBins) + tIdx)];

      if (weight > 0) {
        float localReValue = ((rIdx * rScale) - rMax);
        float theta = (tIdx * radInc);
        linesWithWeights.push_back(std::make_pair(cv::Vec2f(theta, localReValue), weight));
      }
    }
  }

  std::sort(
    linesWithWeights.begin(),
    linesWithWeights.end(),
    [](const std::pair<cv::Vec2f, int> &a, const std::pair<cv::Vec2f, int> &b) {
      return a.second > b.second;
    }
  );

  for (int i = 0; i < std::min(threshold, static_cast<int>(linesWithWeights.size())); i++) {

    cv::Vec2f lineParams = linesWithWeights[i].first;
    float theta = lineParams[0];
    float r = lineParams[1];

    double cosTheta = cos(theta);
    double sinTheta = sin(theta);

    double xValue = (xCenter - (r * cosTheta));
    double yValue = (yCenter - (r * sinTheta));
    double alpha = 1000;

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

  cv::imwrite(outputFileName, imgColor);
}

int main(int argc, char **argv) {

  if (argc < 3) {
    printf("!WARNING [usage: ./hough <pgm-image> <threshold>]\n");
    return EXIT_FAILURE;
  }

  PGMImage inImg(argv[1]);

  int threshold = strtol(argv[2], NULL, 10);

  int *cpuht;
  int w = inImg.x_dim;
  int h = inImg.y_dim;

  CPU_HoughTran(inImg.pixels, w, h, &cpuht);

  float *pcCos = (float*) malloc(sizeof(float) * degreeBins);
  float *pcSin = (float*) malloc(sizeof(float) * degreeBins);
  float rad = 0;

  for (int i = 0; i < degreeBins; i++) {
    pcCos[i] = cos(rad);
    pcSin[i] = sin(rad);
    rad += radInc;
  }

  float rMax = sqrt(1.0 * w * w + 1.0 * h * h) / 2;
  float rScale = ((2 * rMax) / rBins);

  cudaMemcpyToSymbol(d_Cos, pcCos, sizeof(float) * degreeBins);
  cudaMemcpyToSymbol(d_Sin, pcSin, sizeof(float) * degreeBins);

  unsigned char *d_in, *h_in;
  int *d_hough, *h_hough;

  h_in = inImg.pixels;

  h_hough = (int*) malloc(degreeBins * rBins * sizeof(int));

  cudaMalloc((void **) &d_in, (sizeof(unsigned char) * w * h));
  cudaMalloc((void **) &d_hough, (sizeof(int) * degreeBins * rBins));
  cudaMemcpy(d_in, h_in, (sizeof(unsigned char) * w * h), cudaMemcpyHostToDevice);
  cudaMemset(d_hough, 0, (sizeof(int) * degreeBins * rBins));

  int blockNum = ceil((w * h) / 256);

  cudaEvent_t start, stop;
  float elapsedTime;

  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start, 0);

  GPU_HoughTran<<<blockNum, 256>>>(d_in, w, h, d_hough, rMax, rScale);

  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);

  cudaEventElapsedTime(&elapsedTime, start, stop);

  cudaMemcpy(h_hough, d_hough, (sizeof(int) * degreeBins * rBins), cudaMemcpyDeviceToHost);

  for (int i = 0; i < (degreeBins * rBins); i++) {
    if (cpuht[i] != h_hough[i]) {
      printf("➢ calculation mismatch at: %i %i %i\n", i, cpuht[i], h_hough[i]);
    }
  }

  printf("✓ done\n");
  printf("-> kernel execution time: %f ms\n", elapsedTime);
  drawAndSaveLines("results/sharedOutput.jpg", inImg.pixels, w, h, h_hough, rScale, rMax, threshold);

  cudaFree(d_in);
  cudaFree(d_hough);
  cudaFree(d_Cos);
  cudaFree(d_Sin);

  free(h_hough);

  delete[] cpuht;

  return 0;

}