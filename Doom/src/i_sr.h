#ifndef __I_SR__
#define __I_SR__

#include "doomtype.h"

#define I_SR_FRAME_FLAG_MODEL    0x1u
#define I_SR_FRAME_FLAG_FALLBACK 0x2u

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
