// Mouth CNN inference (float), port of tools/train.py MouthNet with BatchNorm folded
// (weights generated into mouth_weights.h by tools/export_c.py).
// Conv3x3 (pad 1) + bias + ReLU, fused 2x2 max pool where the net pools, global mean, linear,
// sigmoid. Activations are HWC; two scratch buffers ping-pong between the layers.
#pragma once

#include <stddef.h>
#include <stdint.h>

namespace mouth {

// Copy the conv weights to memory from alloc (internal RAM on the chip); false if some did not fit
// (those layers keep reading flash). Optional.
bool init(void* (*alloc)(size_t));

int inputWidth();
int inputHeight();
int outputCount();
const char* channelName(int i);

// Floats needed for each of the two scratch buffers.
void scratchSizes(size_t* a, size_t* b);

// img: inputWidth() x inputHeight() grey bytes; out: outputCount() values 0..1.
void run(const uint8_t* img, float* bufA, float* bufB, float* out);

// 4x4 box average: 320x240 grey frame -> 80x60 (same as cv2.INTER_AREA for an exact 4x).
void downscale4(const uint8_t* src, int srcW, int srcH, uint8_t* dst);

}  // namespace mouth
