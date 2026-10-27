#ifndef __I_SR__
#define __I_SR__

#include "doomtype.h"
#include <stdint.h>

typedef struct { uint32_t x, y, w, h; } sr_roi_t;
boolean I_SR_ROIMode(void);
void I_SR_ClearROIs(void);
void I_SR_AddROI(int x, int y, int w, int h);
int I_SR_GetROIs(const sr_roi_t **regions);
const char *I_SR_ModeLabel(void);

#define I_SR_FRAME_FLAG_MODEL    0x1u
#define I_SR_FRAME_FLAG_FALLBACK 0x2u
#define I_SR_FRAME_FLAG_ROI 0x4u
#define I_SR_FRAME_FLAG_BACKGROUND 0x8u
#define I_SR_FRAME_FLAG_BUDGET 0x10u

void I_SR_Init(void);
void I_SR_Shutdown(void);

boolean I_SR_IsAvailable(void);
unsigned int I_SR_GetLastFrameFlags(void);

boolean I_SR_ProcessFrame(const void *pixels,
                          int width,
                          int height,
                          int pitch,
                          const byte **output_pixels,
                          int *output_width,
                          int *output_height,
                          int *output_pitch);

#endif
