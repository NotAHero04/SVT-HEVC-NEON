/* 
* Copyright(c) 2018 Intel Corporation
* SPDX - License - Identifier: BSD - 2 - Clause - Patent
*/

#include "EbDefinitions.h"
#include "../../../simde/simde/x86/avx2.h"
#include "EbIntrinMacros_SSE2.h"
#include "EbIntraPrediction_AVX2.h"

#ifndef simde_mm256_setr_m128i
#define simde_mm256_setr_m128i(/* simde__m128i */ hi, /* simde__m128i */ lo) \
    simde_mm256_insertf128_si256(simde_mm256_castsi128_si256(lo), (hi), 0x1)
#endif

#define MACRO_VERTICAL_LUMA_4(A, B, C) \
    *(EB_U32*)predictionPtr = simde_mm_cvtsi128_si32(simde_mm_or_si128(simde_mm_and_si128(A, B), C)); \
    A = simde_mm_srli_si128(A, 1); \
    *(EB_U32*)(predictionPtr + pStride) = simde_mm_cvtsi128_si32(simde_mm_or_si128(simde_mm_and_si128(A, B), C)); \
    A = simde_mm_srli_si128(A, 1);

#define simde_mm256_set_m128i(/* simde__m128i */ hi, /* simde__m128i */ lo) \
    simde_mm256_insertf128_si256(simde_mm256_castsi128_si256(lo), (hi), 0x1)

static void simde_mm_storeh_epi64(simde__m128i * p, simde__m128i x)
{
  simde_mm_storeh_pd((double *)p, simde_mm_castsi128_pd(x));
}

EB_EXTERN void IntraModePlanar_AVX2_INTRIN(
  const EB_U32   size,                       //input parameter, denotes the size of the current PU
  EB_U8         *refSamples,                 //input parameter, pointer to the reference samples
  EB_U8         *predictionPtr,              //output parameter, pointer to the prediction
  const EB_U32   predictionBufferStride,     //input parameter, denotes the stride for the prediction ptr
  const EB_BOOL  skip)                       //skip half rows
{
  EB_U32 topOffset = (size << 1) + 1;
  EB_U8  topRightPel, bottomLeftPel;
  EB_U32 x, y;
  simde__m128i left, delta0, delta1, a0, a1, a2, a3, a4;
  simde__m256i lleft, ddelta0, ddelta1, aa0, aa1, aa2, aa3, aa4, cc0, cc1;

  // --------- Reference Samples Structure ---------
  // (refSamples are similar as vertical mode)
  // refSamples[0]        = Left[0]
  // refSamples[1]        = Left[1]
  // ...
  // refSamples[size]     = Left[size]
  // ... (arbitrary value)
  // refSamples[2*size+1] = Top[0]
  // refSamples[2*size+2] = Top[1]
  // ...
  // refSamples[3*size+1] = Top[size]
  // -----------------------------------------------

  // Get above and left reference samples
  topRightPel = refSamples[topOffset + size];
  bottomLeftPel = refSamples[size];

  // Generate prediction
  if (size == 4) {
    a0 = simde_mm_set1_epi8(topRightPel);
    a1 = simde_mm_set1_epi16(bottomLeftPel);
    left = simde_mm_cvtsi32_si128(*(EB_U32 *)refSamples); // leftPel
    a2 = simde_mm_cvtsi32_si128(*(EB_U32 *)(refSamples + topOffset)); // topPel
    delta0 = simde_mm_unpacklo_epi8(a2, simde_mm_setzero_si128());
    delta0 = simde_mm_sub_epi16(a1, delta0);
    a2 = simde_mm_unpacklo_epi8(a2, a0);
    a2 = simde_mm_maddubs_epi16(a2, simde_mm_setr_epi8(4, 1, 4, 2, 4, 3, 4, 4, 0, 0, 0, 0, 0, 0, 0, 0)); // (x + 1)*topRightPel + size * refSamples[topOffset + x]
    a2 = simde_mm_add_epi16(a2, simde_mm_set1_epi16(4)); // (x + 1)*topRightPel + size * refSamples[topOffset + x] + size
    if (skip) {
      a2 = simde_mm_add_epi16(a2, delta0);
      delta0 = simde_mm_slli_epi16(delta0, 1);
      a4 = simde_mm_shuffle_epi8(left, simde_mm_setzero_si128());
      a0 = simde_mm_add_epi16(a2, simde_mm_maddubs_epi16(a4, simde_mm_setr_epi8(3, 0, 2, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)));
      a0 = simde_mm_srai_epi16(a0, 3);
      a0 = simde_mm_packus_epi16(a0, a0);
      *(EB_U32 *)predictionPtr = simde_mm_cvtsi128_si32(a0);
      a2 = simde_mm_add_epi16(a2, delta0);
      left = simde_mm_srli_si128(left, 2);
      a4 = simde_mm_shuffle_epi8(left, simde_mm_setzero_si128());
      a0 = simde_mm_add_epi16(a2, simde_mm_maddubs_epi16(a4, simde_mm_setr_epi8(3, 0, 2, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)));
      a0 = simde_mm_srai_epi16(a0, 3);
      a0 = simde_mm_packus_epi16(a0, a0);
      *(EB_U32 *)(predictionPtr + 2 * predictionBufferStride) = simde_mm_cvtsi128_si32(a0);
    }
    else {
      for (y = 0; y < 4; y++) {
        a2 = simde_mm_add_epi16(a2, delta0);
        a4 = simde_mm_shuffle_epi8(left, simde_mm_setzero_si128());
        a0 = simde_mm_add_epi16(a2, simde_mm_maddubs_epi16(a4, simde_mm_setr_epi8(3, 0, 2, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)));
        a0 = simde_mm_srai_epi16(a0, 3);
        a0 = simde_mm_packus_epi16(a0, a0);
        *(EB_U32 *)predictionPtr = simde_mm_cvtsi128_si32(a0);
        predictionPtr += predictionBufferStride;
        left = simde_mm_srli_si128(left, 1);
      }
    }
  }
  else if (size == 8) {
    a0 = simde_mm_set1_epi8(topRightPel);
    a1 = simde_mm_set1_epi16(bottomLeftPel);
    left = simde_mm_loadl_epi64((simde__m128i *)refSamples); // leftPel
    a2 = simde_mm_loadl_epi64((simde__m128i *)(refSamples + topOffset)); // topPel
    delta0 = simde_mm_unpacklo_epi8(a2, simde_mm_setzero_si128());
    delta0 = simde_mm_sub_epi16(a1, delta0);
    a2 = simde_mm_unpacklo_epi8(a2, a0);
    a2 = simde_mm_maddubs_epi16(a2, simde_mm_setr_epi8(8, 1, 8, 2, 8, 3, 8, 4, 8, 5, 8, 6, 8, 7, 8, 8)); // (x + 1)*topRightPel + size * refSamples[topOffset + x]
    a2 = simde_mm_add_epi16(a2, simde_mm_set1_epi16(8)); // (x + 1)*topRightPel + size * refSamples[topOffset + x] + size
    if (skip) {
      a2 = simde_mm_add_epi16(a2, delta0);
      delta0 = simde_mm_slli_epi16(delta0, 1);
      for (y = 0; y < 8; y += 2) {
        a4 = simde_mm_shuffle_epi8(left, simde_mm_setzero_si128());
        a0 = simde_mm_add_epi16(a2, simde_mm_maddubs_epi16(a4, simde_mm_setr_epi8(7, 0, 6, 0, 5, 0, 4, 0, 3, 0, 2, 0, 1, 0, 0, 0)));
        a0 = simde_mm_srai_epi16(a0, 4);
        a0 = simde_mm_packus_epi16(a0, a0);
        simde_mm_storel_epi64((simde__m128i *)predictionPtr, a0);
        predictionPtr += 2 * predictionBufferStride;
        a2 = simde_mm_add_epi16(a2, delta0);
        left = simde_mm_srli_si128(left, 2);
      }
    }
    else {
      for (y = 0; y < 8; y++) {
        a2 = simde_mm_add_epi16(a2, delta0);
        a4 = simde_mm_shuffle_epi8(left, simde_mm_setzero_si128());
        a0 = simde_mm_add_epi16(a2, simde_mm_maddubs_epi16(a4, simde_mm_setr_epi8(7, 0, 6, 0, 5, 0, 4, 0, 3, 0, 2, 0, 1, 0, 0, 0)));
        a0 = simde_mm_srai_epi16(a0, 4);
        a0 = simde_mm_packus_epi16(a0, a0);
        simde_mm_storel_epi64((simde__m128i *)predictionPtr, a0);
        predictionPtr += predictionBufferStride;
        left = simde_mm_srli_si128(left, 1);
      }
    }
  }
  else if (size == 16) {
    a0 = simde_mm_set1_epi8(topRightPel);
    a1 = simde_mm_set1_epi16(bottomLeftPel);
    left = simde_mm_loadu_si128((simde__m128i *)refSamples); // leftPel
    a2 = simde_mm_loadu_si128((simde__m128i *)(refSamples + topOffset)); // topPel
    delta0 = simde_mm_unpacklo_epi8(a2, simde_mm_setzero_si128());
    delta1 = simde_mm_unpackhi_epi8(a2, simde_mm_setzero_si128());
    delta0 = simde_mm_sub_epi16(a1, delta0);
    delta1 = simde_mm_sub_epi16(a1, delta1);
    a3 = simde_mm_unpackhi_epi8(a2, a0);
    a2 = simde_mm_unpacklo_epi8(a2, a0);
    a2 = simde_mm_maddubs_epi16(a2, simde_mm_setr_epi8(16, 1, 16, 2, 16, 3, 16, 4, 16, 5, 16, 6, 16, 7, 16, 8)); // (x + 1)*topRightPel + size * refSamples[topOffset + x]
    a3 = simde_mm_maddubs_epi16(a3, simde_mm_setr_epi8(16, 9, 16, 10, 16, 11, 16, 12, 16, 13, 16, 14, 16, 15, 16, 16)); // (x + 1)*topRightPel + size * refSamples[topOffset + x]
    a2 = simde_mm_add_epi16(a2, simde_mm_set1_epi16(16)); // (x + 1)*topRightPel + size * refSamples[topOffset + x] + size
    a3 = simde_mm_add_epi16(a3, simde_mm_set1_epi16(16));
    if (skip) {
      a2 = simde_mm_add_epi16(a2, delta0);
      a3 = simde_mm_add_epi16(a3, delta1);
      delta0 = simde_mm_slli_epi16(delta0, 1);
      delta1 = simde_mm_slli_epi16(delta1, 1);
      for (y = 0; y < 16; y += 2) {
        a4 = simde_mm_shuffle_epi8(left, simde_mm_setzero_si128());
        a0 = simde_mm_add_epi16(a2, simde_mm_maddubs_epi16(a4, simde_mm_setr_epi8(15, 0, 14, 0, 13, 0, 12, 0, 11, 0, 10, 0, 9, 0, 8, 0)));
        a1 = simde_mm_add_epi16(a3, simde_mm_maddubs_epi16(a4, simde_mm_setr_epi8(7, 0, 6, 0, 5, 0, 4, 0, 3, 0, 2, 0, 1, 0, 0, 0)));
        a0 = simde_mm_srai_epi16(a0, 5);
        a1 = simde_mm_srai_epi16(a1, 5);
        a0 = simde_mm_packus_epi16(a0, a1);
        simde_mm_storeu_si128((simde__m128i *)predictionPtr, a0);
        predictionPtr += 2 * predictionBufferStride;
        a2 = simde_mm_add_epi16(a2, delta0);
        a3 = simde_mm_add_epi16(a3, delta1);
        left = simde_mm_srli_si128(left, 2);
      }
    }
    else {
      for (y = 0; y < 16; y++) {
        a2 = simde_mm_add_epi16(a2, delta0);
        a3 = simde_mm_add_epi16(a3, delta1);
        a4 = simde_mm_shuffle_epi8(left, simde_mm_setzero_si128());
        a0 = simde_mm_add_epi16(a2, simde_mm_maddubs_epi16(a4, simde_mm_setr_epi8(15, 0, 14, 0, 13, 0, 12, 0, 11, 0, 10, 0, 9, 0, 8, 0)));
        a1 = simde_mm_add_epi16(a3, simde_mm_maddubs_epi16(a4, simde_mm_setr_epi8(7, 0, 6, 0, 5, 0, 4, 0, 3, 0, 2, 0, 1, 0, 0, 0)));
        a0 = simde_mm_srai_epi16(a0, 5);
        a1 = simde_mm_srai_epi16(a1, 5);
        a0 = simde_mm_packus_epi16(a0, a1);
        simde_mm_storeu_si128((simde__m128i *)predictionPtr, a0);
        predictionPtr += predictionBufferStride;
        left = simde_mm_srli_si128(left, 1);
      }
    }
  }
  else if (size == 32) {
    aa0 = simde_mm256_set1_epi8(topRightPel);
    aa1 = simde_mm256_set1_epi16(bottomLeftPel);
    //lleft = simde_mm256_loadu_si256((simde__m256i *)refSamples); // leftPel
    //lleft = simde_mm256_permute4x64_epi64(lleft, 0x44);
    aa2 = simde_mm256_loadu_si256((simde__m256i *)(refSamples + topOffset)); // topPel
    ddelta0 = simde_mm256_unpacklo_epi8(aa2, simde_mm256_setzero_si256());
    ddelta1 = simde_mm256_unpackhi_epi8(aa2, simde_mm256_setzero_si256());
    ddelta0 = simde_mm256_sub_epi16(aa1, ddelta0);
    ddelta1 = simde_mm256_sub_epi16(aa1, ddelta1);
    aa3 = simde_mm256_unpackhi_epi8(aa2, aa0);
    aa2 = simde_mm256_unpacklo_epi8(aa2, aa0);
    aa2 = simde_mm256_maddubs_epi16(aa2, simde_mm256_setr_epi8(32, 1, 32, 2, 32, 3, 32, 4, 32, 5, 32, 6, 32, 7, 32, 8, 32, 17, 32, 18, 32, 19, 32, 20, 32, 21, 32, 22, 32, 23, 32, 24)); // (x + 1)*topRightPel + size * refSamples[topOffset + x]
    aa3 = simde_mm256_maddubs_epi16(aa3, simde_mm256_setr_epi8(32, 9, 32, 10, 32, 11, 32, 12, 32, 13, 32, 14, 32, 15, 32, 16, 32, 25, 32, 26, 32, 27, 32, 28, 32, 29, 32, 30, 32, 31, 32, 32)); // (x + 1)*topRightPel + size * refSamples[topOffset + x]
    aa2 = simde_mm256_add_epi16(aa2, simde_mm256_set1_epi16(32)); // (x + 1)*topRightPel + size * refSamples[topOffset + x] + size
    aa3 = simde_mm256_add_epi16(aa3, simde_mm256_set1_epi16(32));
    cc0 = simde_mm256_setr_epi8(31, 0, 30, 0, 29, 0, 28, 0, 27, 0, 26, 0, 25, 0, 24, 0, 15, 0, 14, 0, 13, 0, 12, 0, 11, 0, 10, 0, 9, 0, 8, 0);
    cc1 = simde_mm256_setr_epi8(23, 0, 22, 0, 21, 0, 20, 0, 19, 0, 18, 0, 17, 0, 16, 0, 7, 0, 6, 0, 5, 0, 4, 0, 3, 0, 2, 0, 1, 0, 0, 0);
    if (skip) {
      lleft = simde_mm256_loadu_si256((simde__m256i *)refSamples); // leftPel
      lleft = simde_mm256_shuffle_epi8(lleft, simde_mm256_setr_epi8(0, 2, 4, 6, 8, 10, 12, 14, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 4, 6, 8, 10, 12, 14, 0, 0, 0, 0, 0, 0, 0, 0));
      lleft = simde_mm256_permute4x64_epi64(lleft, 0x88);
      aa2 = simde_mm256_add_epi16(aa2, ddelta0);
      aa3 = simde_mm256_add_epi16(aa3, ddelta1);
      ddelta0 = simde_mm256_slli_epi16(ddelta0, 1);
      ddelta1 = simde_mm256_slli_epi16(ddelta1, 1);
      for (y = 0; y < 16; y++) {
        aa4 = simde_mm256_shuffle_epi8(lleft, simde_mm256_setzero_si256());
        aa0 = simde_mm256_maddubs_epi16(aa4, cc0);
        aa1 = simde_mm256_maddubs_epi16(aa4, cc1);
        aa0 = simde_mm256_add_epi16(aa0, aa2);
        aa1 = simde_mm256_add_epi16(aa1, aa3);
        aa0 = simde_mm256_srai_epi16(aa0, 6);
        aa1 = simde_mm256_srai_epi16(aa1, 6);
        aa0 = simde_mm256_packus_epi16(aa0, aa1);
        simde_mm256_storeu_si256((simde__m256i *)predictionPtr, aa0);
        predictionPtr += 2 * predictionBufferStride;
        aa2 = simde_mm256_add_epi16(aa2, ddelta0);
        aa3 = simde_mm256_add_epi16(aa3, ddelta1);
        lleft = simde_mm256_srli_si256(lleft, 1);
      }
    }
    else {
      for (x = 0; x < 2; x++) {
        lleft = simde_mm256_loadu_si256((simde__m256i *)refSamples); // leftPel
        lleft = simde_mm256_permute4x64_epi64(lleft, 0x44);
        refSamples += 16;
        for (y = 0; y < 16; y++) {
          aa2 = simde_mm256_add_epi16(aa2, ddelta0);
          aa3 = simde_mm256_add_epi16(aa3, ddelta1);
          aa4 = simde_mm256_shuffle_epi8(lleft, simde_mm256_setzero_si256());
          aa0 = simde_mm256_maddubs_epi16(aa4, cc0);
          aa1 = simde_mm256_maddubs_epi16(aa4, cc1);
          aa0 = simde_mm256_add_epi16(aa0, aa2);
          aa1 = simde_mm256_add_epi16(aa1, aa3);
          aa0 = simde_mm256_srai_epi16(aa0, 6);
          aa1 = simde_mm256_srai_epi16(aa1, 6);
          aa0 = simde_mm256_packus_epi16(aa0, aa1);
          simde_mm256_storeu_si256((simde__m256i *)predictionPtr, aa0);
          predictionPtr += predictionBufferStride;
          lleft = simde_mm256_srli_si256(lleft, 1);
        }
      }
    }
  }
}

EB_EXTERN void IntraModeAngular_Vertical_Kernel_AVX2_INTRIN(
  EB_U32         size,                       //input parameter, denotes the size of the current PU
  EB_U8         *refSampMain,                //input parameter, pointer to the reference samples
  EB_U8         *predictionPtr,              //output parameter, pointer to the prediction
  EB_U32   predictionBufferStride,     //input parameter, denotes the stride for the prediction ptr
  const EB_BOOL  skip,
  EB_S32   intraPredAngle)
{
  EB_U32 rowIndex;
  EB_U32 height = size;
  EB_S32 deltaSum = intraPredAngle;
  EB_S32 deltaInt;
  EB_U32 deltaFract;
  simde__m128i top0, top1, top2, sum0, sum1, a0, a1;
  simde__m256i ttop0, ttop1, ttop2, ssum0, ssum1, aa0;

  // --------- Reference Samples Structure ---------
  // refSampMain[-size+1] to refSampMain[-1] must be prepared (from bottom to top) for mode 19 to 25 (not required for mode 27 to 33)
  // refSampMain[0]      = TopLeft[0]
  // refSampMain[1]      = Top[0]
  // refSampMain[2]      = Top[1]
  // ...
  // refSampMain[size]   = Top[size-1]
  // refSampMain[size+1] = Top[size]     for mode 27 to 33 (not required for mode 19 to 25)
  // ...
  // refSampMain[2*size] = Top[2*size-1] for mode 27 to 33 (not required for mode 19 to 25)
  // -----------------------------------------------

  // Compute the prediction
  refSampMain += 1; // top0 sample
  if(skip) {
    height >>= 1;
    predictionBufferStride <<= 1;
    intraPredAngle <<= 1;
  }

  if(size == 4) {
    for(rowIndex=0; rowIndex<height; rowIndex+=2) {
      deltaInt   = deltaSum >> 5;
      deltaFract = deltaSum & 31;
      a0 = simde_mm_unpacklo_epi8(simde_mm_cvtsi32_si128(32 - deltaFract), simde_mm_cvtsi32_si128(deltaFract));
      top0 = simde_mm_loadl_epi64((simde__m128i *)(refSampMain+deltaInt));
      deltaSum += intraPredAngle;
      deltaInt   = deltaSum >> 5;
      deltaFract = deltaSum & 31;
      a1 = simde_mm_unpacklo_epi8(simde_mm_cvtsi32_si128(32 - deltaFract), simde_mm_cvtsi32_si128(deltaFract));
      a0 = simde_mm_unpacklo_epi16(a0, a1);
      a0 = simde_mm_shuffle_epi8(a0, simde_mm_setr_epi8(0, 1, 0, 1, 0, 1, 0, 1, 2, 3, 2, 3, 2, 3, 2, 3));
      top0 = simde_mm_castps_si128(simde_mm_loadh_pi(simde_mm_castsi128_ps(top0), (simde__m64 *)(refSampMain+deltaInt)));
      top0 = simde_mm_shuffle_epi8(top0, simde_mm_setr_epi8(0, 1, 1, 2, 2, 3, 3, 4, 8, 9, 9, 10, 10, 11, 11, 12));
      sum0 = simde_mm_add_epi16(simde_mm_set1_epi16(16), simde_mm_maddubs_epi16(top0, a0));
      sum0 = simde_mm_srai_epi16(sum0, 5);
      sum0 = simde_mm_packus_epi16(sum0, sum0);
      *(EB_U32 *)predictionPtr = simde_mm_cvtsi128_si32(sum0);
      *(EB_U32 *)(predictionPtr+predictionBufferStride) = simde_mm_cvtsi128_si32(simde_mm_srli_si128(sum0, 4));
      predictionPtr += 2 * predictionBufferStride;
      deltaSum += intraPredAngle;
    }
  }
  else if(size == 8) {
    for(rowIndex=0; rowIndex<height; rowIndex++) {
      deltaInt   = deltaSum >> 5;
      deltaFract = deltaSum & 31;
      a0 = simde_mm_unpacklo_epi8(simde_mm_cvtsi32_si128(32 - deltaFract), simde_mm_cvtsi32_si128(deltaFract));
      a0 = simde_mm_shuffle_epi8(a0, simde_mm_setr_epi8(0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1));
      top0 = simde_mm_loadu_si128((simde__m128i *)(refSampMain+deltaInt));
      top0 = simde_mm_shuffle_epi8(top0, simde_mm_setr_epi8(0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8));
      sum0 = simde_mm_add_epi16(simde_mm_set1_epi16(16), simde_mm_maddubs_epi16(top0, a0));
      sum0 = simde_mm_srai_epi16(sum0, 5);
      sum0 = simde_mm_packus_epi16(sum0, sum0);
      simde_mm_storel_epi64((simde__m128i *)predictionPtr, sum0);
      predictionPtr += predictionBufferStride;
      deltaSum += intraPredAngle;
    }
  }
  else if(size == 16) {
    for(rowIndex=0; rowIndex<height; rowIndex++) {
      deltaInt   = deltaSum >> 5;
      deltaFract = deltaSum & 31;
      a0 = simde_mm_unpacklo_epi8(simde_mm_cvtsi32_si128(32 - deltaFract), simde_mm_cvtsi32_si128(deltaFract));
      a0 = simde_mm_shuffle_epi8(a0, simde_mm_setr_epi8(0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1));
      top0 = simde_mm_loadu_si128((simde__m128i *)(refSampMain+deltaInt));
      top1 = simde_mm_loadu_si128((simde__m128i *)(refSampMain+deltaInt+1));
      top2 = simde_mm_unpacklo_epi8(top0, top1);
      top0 = simde_mm_unpackhi_epi8(top0, top1);
      sum0 = simde_mm_add_epi16(simde_mm_set1_epi16(16), simde_mm_maddubs_epi16(top2, a0));
      sum1 = simde_mm_add_epi16(simde_mm_set1_epi16(16), simde_mm_maddubs_epi16(top0, a0));
      sum0 = simde_mm_srai_epi16(sum0, 5);
      sum1 = simde_mm_srai_epi16(sum1, 5);
      sum0 = simde_mm_packus_epi16(sum0, sum1);
      simde_mm_storeu_si128((simde__m128i *)predictionPtr, sum0);
      predictionPtr += predictionBufferStride;
      deltaSum += intraPredAngle;
    }
  }
  else { // size == 32
    for(rowIndex=0; rowIndex<height; rowIndex++) {
      deltaInt   = deltaSum >> 5;
      deltaFract = deltaSum & 31;
      a0 = simde_mm_unpacklo_epi8(simde_mm_cvtsi32_si128(32 - deltaFract), simde_mm_cvtsi32_si128(deltaFract));
      a0 = simde_mm_shuffle_epi8(a0, simde_mm_setr_epi8(0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1));
      aa0 = simde_mm256_setr_m128i(a0, a0);
      ttop0 = simde_mm256_loadu_si256((simde__m256i *)(refSampMain + deltaInt));
      ttop1 = simde_mm256_loadu_si256((simde__m256i *)(refSampMain + deltaInt+1));
      ttop2 = simde_mm256_unpacklo_epi8(ttop0, ttop1);
      ttop0 = simde_mm256_unpackhi_epi8(ttop0, ttop1);
      ssum0 = simde_mm256_add_epi16(simde_mm256_set1_epi16(16), simde_mm256_maddubs_epi16(ttop2, aa0));
      ssum1 = simde_mm256_add_epi16(simde_mm256_set1_epi16(16), simde_mm256_maddubs_epi16(ttop0, aa0));
      ssum0 = simde_mm256_srai_epi16(ssum0, 5);
      ssum1 = simde_mm256_srai_epi16(ssum1, 5);
      ssum0 = simde_mm256_packus_epi16(ssum0, ssum1);
      simde_mm256_storeu_si256((simde__m256i *)predictionPtr, ssum0);
      predictionPtr += predictionBufferStride;
      deltaSum += intraPredAngle;
    }
  }
}

EB_EXTERN void IntraModeAngular_Horizontal_Kernel_AVX2_INTRIN(
  EB_U32         size,                       //input parameter, denotes the size of the current PU
  EB_U8         *refSampMain,                //input parameter, pointer to the reference samples
  EB_U8         *predictionPtr,              //output parameter, pointer to the prediction
  EB_U32         predictionBufferStride,     //input parameter, denotes the stride for the prediction ptr
  const EB_BOOL  skip,
  EB_S32         intraPredAngle)
{
  EB_U32 rowIndex, colIndex;
  //EB_U32 rowStride = skip ? 2 : 1;
  EB_S32 deltaSum = 0;
  EB_S32 deltaInt;
  EB_U32 deltaFract;
  simde__m128i top0, top1, top2, sum0, sum1, a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11;
  simde__m256i ttop0, ttop1, ttop2, ssum0, ssum1, aa0;
  //simde__m256i aa1, aa2, aa3, aa4, aa5, aa6, aa7, aa8, aa9, aa10, aa11;
  EB_U8 tempBuf[32 * 32];
  EB_U8 *p = tempBuf;

  // --------- Reference Samples Structure ---------
  // refSampMain[-size+1] to refSampMain[-1] must be prepared (from right to left) for mode 11 to 17 (not required for mode 3 to 9)
  // refSampMain[0]      = TopLeft[0]
  // refSampMain[1]      = Left[0]
  // refSampMain[2]      = Left[1]
  // ...
  // refSampMain[size]   = Left[size-1]
  // refSampMain[size+1] = Left[size]     for mode 3 to 9 (not required for mode 11 to 17)
  // ...
  // refSampMain[2*size] = Left[2*size-1] for mode 3 to 9 (not required for mode 11 to 17)
  // -----------------------------------------------

  // Compute the prediction
  refSampMain += 1; // left sample

  if(skip) {
    predictionBufferStride <<= 1;
    if(size == 4) {
      deltaSum += intraPredAngle;
      deltaInt   = deltaSum >> 5;
      deltaFract = deltaSum & 31;
      a0 = simde_mm_unpacklo_epi8(simde_mm_cvtsi32_si128(32 - deltaFract), simde_mm_cvtsi32_si128(deltaFract));
      top0 = simde_mm_cvtsi32_si128(*(EB_U32 *)(refSampMain+deltaInt));

      deltaSum += intraPredAngle;
      deltaInt   = deltaSum >> 5;
      deltaFract = deltaSum & 31;
      a1 = simde_mm_unpacklo_epi8(simde_mm_cvtsi32_si128(32 - deltaFract), simde_mm_cvtsi32_si128(deltaFract));
      a0 = simde_mm_unpacklo_epi16(a0, a1);
      top0 = simde_mm_unpacklo_epi32(top0, simde_mm_cvtsi32_si128(*(EB_U32 *)(refSampMain+deltaInt)));

      deltaSum += intraPredAngle;
      deltaInt   = deltaSum >> 5;
      deltaFract = deltaSum & 31;
      a2 = simde_mm_unpacklo_epi8(simde_mm_cvtsi32_si128(32 - deltaFract), simde_mm_cvtsi32_si128(deltaFract));
      a4 = simde_mm_cvtsi32_si128(*(EB_U32 *)(refSampMain+deltaInt));

      deltaSum += intraPredAngle;
      deltaInt   = deltaSum >> 5;
      deltaFract = deltaSum & 31;
      a3 = simde_mm_unpacklo_epi8(simde_mm_cvtsi32_si128(32 - deltaFract), simde_mm_cvtsi32_si128(deltaFract));
      a2 = simde_mm_unpacklo_epi16(a2, a3);
      a0 = simde_mm_unpacklo_epi32(a0, a2);
      a0 = simde_mm_unpacklo_epi16(a0, a0);
      a4 = simde_mm_unpacklo_epi32(a4, simde_mm_cvtsi32_si128(*(EB_U32 *)(refSampMain+deltaInt)));

      top0 = simde_mm_unpacklo_epi64(top0, a4);
      sum0 = simde_mm_add_epi16(simde_mm_set1_epi16(16), simde_mm_maddubs_epi16(top0, a0));
      sum0 = simde_mm_srai_epi16(sum0, 5);
      sum0 = simde_mm_packus_epi16(sum0, sum0);
      sum0 = simde_mm_shuffle_epi8(sum0, simde_mm_setr_epi8(0, 2, 4, 6, 1, 3, 5, 7, 2, 6, 10, 14, 3, 7, 11, 15));
      *(EB_U32 *)predictionPtr = simde_mm_cvtsi128_si32(sum0); sum0 = simde_mm_srli_si128(sum0, 4); predictionPtr += predictionBufferStride;
      *(EB_U32 *)predictionPtr = simde_mm_cvtsi128_si32(sum0);
    }
    else if(size == 8) {
      for(colIndex=0; colIndex<size; colIndex++) {
        deltaSum += intraPredAngle;
        deltaInt   = deltaSum >> 5;
        deltaFract = deltaSum & 31;
        a0 = simde_mm_unpacklo_epi8(simde_mm_cvtsi32_si128(32 - deltaFract), simde_mm_cvtsi32_si128(deltaFract));
        a0 = simde_mm_shuffle_epi8(a0, simde_mm_setr_epi8(0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1));
        top0 = simde_mm_loadl_epi64((simde__m128i *)(refSampMain+deltaInt));
        sum0 = simde_mm_add_epi16(simde_mm_set1_epi16(16), simde_mm_maddubs_epi16(top0, a0));
        sum0 = simde_mm_srai_epi16(sum0, 5);
        sum0 = simde_mm_packus_epi16(sum0, sum0);
        *(EB_U32 *)p = simde_mm_cvtsi128_si32(sum0);
        p += 4;
      }
      a0 = simde_mm_loadu_si128((simde__m128i *)tempBuf);
      a0 = simde_mm_shuffle_epi8(a0, simde_mm_setr_epi8(0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15));
      a1 = simde_mm_loadu_si128((simde__m128i *)(tempBuf+16));
      a1 = simde_mm_shuffle_epi8(a1, simde_mm_setr_epi8(0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15));
      a2  = simde_mm_unpackhi_epi32(a0, a1);
      a0  = simde_mm_unpacklo_epi32(a0, a1);
      a1  = simde_mm_unpackhi_epi64(a0, a2);
      a0  = simde_mm_unpacklo_epi64(a0, a2);
      simde_mm_storel_epi64((simde__m128i *)predictionPtr, a0); predictionPtr += predictionBufferStride;
      simde_mm_storel_epi64((simde__m128i *)predictionPtr, a1); predictionPtr += predictionBufferStride;
      simde_mm_storeh_epi64((simde__m128i *)predictionPtr, a0); predictionPtr += predictionBufferStride;
      simde_mm_storeh_epi64((simde__m128i *)predictionPtr, a1);
    }
    else if(size == 16) {
      for(colIndex=0; colIndex<size; colIndex++) {
        deltaSum += intraPredAngle;
        deltaInt   = deltaSum >> 5;
        deltaFract = deltaSum & 31;
        a0 = simde_mm_unpacklo_epi8(simde_mm_cvtsi32_si128(32 - deltaFract), simde_mm_cvtsi32_si128(deltaFract));
        a0 = simde_mm_shuffle_epi8(a0, simde_mm_setr_epi8(0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1));
        top0 = simde_mm_loadu_si128((simde__m128i *)(refSampMain + deltaInt));
        top0 = simde_mm_xor_si128(top0, simde_mm_setzero_si128()); // Redundant code. Visual Studio Express 2013 Release build messes up the order of operands in simde_mm_maddubs_epi16(), and hard to work around.
        sum0 = simde_mm_add_epi16(simde_mm_set1_epi16(16), simde_mm_maddubs_epi16(top0, a0));
        sum0 = simde_mm_srai_epi16(sum0, 5);
        sum0 = simde_mm_packus_epi16(sum0, sum0);
        simde_mm_storel_epi64((simde__m128i *)p, sum0);
        p += 8;
      }
      p = tempBuf;
      a0  = simde_mm_unpacklo_epi8(simde_mm_loadl_epi64((simde__m128i *)(p+0x00)), simde_mm_loadl_epi64((simde__m128i *)(p+0x08)));
      a1  = simde_mm_unpacklo_epi8(simde_mm_loadl_epi64((simde__m128i *)(p+0x10)), simde_mm_loadl_epi64((simde__m128i *)(p+0x18)));
      a2  = simde_mm_unpacklo_epi8(simde_mm_loadl_epi64((simde__m128i *)(p+0x20)), simde_mm_loadl_epi64((simde__m128i *)(p+0x28)));
      a3  = simde_mm_unpacklo_epi8(simde_mm_loadl_epi64((simde__m128i *)(p+0x30)), simde_mm_loadl_epi64((simde__m128i *)(p+0x38)));
      a4  = simde_mm_unpacklo_epi8(simde_mm_loadl_epi64((simde__m128i *)(p+0x40)), simde_mm_loadl_epi64((simde__m128i *)(p+0x48)));
      a5  = simde_mm_unpacklo_epi8(simde_mm_loadl_epi64((simde__m128i *)(p+0x50)), simde_mm_loadl_epi64((simde__m128i *)(p+0x58)));
      a6  = simde_mm_unpacklo_epi8(simde_mm_loadl_epi64((simde__m128i *)(p+0x60)), simde_mm_loadl_epi64((simde__m128i *)(p+0x68)));
      a7  = simde_mm_unpacklo_epi8(simde_mm_loadl_epi64((simde__m128i *)(p+0x70)), simde_mm_loadl_epi64((simde__m128i *)(p+0x78)));

      a8  = simde_mm_unpackhi_epi16(a0, a1);
      a0  = simde_mm_unpacklo_epi16(a0, a1);
      a9  = simde_mm_unpackhi_epi16(a2, a3);
      a2  = simde_mm_unpacklo_epi16(a2, a3);
      a10 = simde_mm_unpackhi_epi16(a4, a5);
      a4  = simde_mm_unpacklo_epi16(a4, a5);
      a11 = simde_mm_unpackhi_epi16(a6, a7);
      a6  = simde_mm_unpacklo_epi16(a6, a7);

      a1  = simde_mm_unpackhi_epi32(a0,  a2);
      a0  = simde_mm_unpacklo_epi32(a0,  a2);
      a3  = simde_mm_unpackhi_epi32(a4,  a6);
      a4  = simde_mm_unpacklo_epi32(a4,  a6);
      a5  = simde_mm_unpackhi_epi32(a8,  a9);
      a8  = simde_mm_unpacklo_epi32(a8,  a9);
      a7  = simde_mm_unpackhi_epi32(a10, a11);
      a10 = simde_mm_unpacklo_epi32(a10, a11);

      a2  = simde_mm_unpackhi_epi64(a0, a4);
      a0  = simde_mm_unpacklo_epi64(a0, a4);
      a6  = simde_mm_unpackhi_epi64(a8, a10);
      a8  = simde_mm_unpacklo_epi64(a8, a10);
      a9  = simde_mm_unpackhi_epi64(a1, a3);
      a1  = simde_mm_unpacklo_epi64(a1, a3);
      a11 = simde_mm_unpackhi_epi64(a5, a7);
      a5  = simde_mm_unpacklo_epi64(a5, a7);

      simde_mm_storeu_si128((simde__m128i *)predictionPtr, a0);  predictionPtr += predictionBufferStride;
      simde_mm_storeu_si128((simde__m128i *)predictionPtr, a2);  predictionPtr += predictionBufferStride;
      simde_mm_storeu_si128((simde__m128i *)predictionPtr, a1);  predictionPtr += predictionBufferStride;
      simde_mm_storeu_si128((simde__m128i *)predictionPtr, a9);  predictionPtr += predictionBufferStride;
      simde_mm_storeu_si128((simde__m128i *)predictionPtr, a8);  predictionPtr += predictionBufferStride;
      simde_mm_storeu_si128((simde__m128i *)predictionPtr, a6);  predictionPtr += predictionBufferStride;
      simde_mm_storeu_si128((simde__m128i *)predictionPtr, a5);  predictionPtr += predictionBufferStride;
      simde_mm_storeu_si128((simde__m128i *)predictionPtr, a11); predictionPtr += predictionBufferStride;
    }
    else { // size == 32
      for(colIndex=0; colIndex<size; colIndex++) {
        deltaSum += intraPredAngle;
        deltaInt   = deltaSum >> 5;
        deltaFract = deltaSum & 31;
        a0 = simde_mm_unpacklo_epi8(simde_mm_cvtsi32_si128(32 - deltaFract), simde_mm_cvtsi32_si128(deltaFract));
        a0 = simde_mm_shuffle_epi8(a0, simde_mm_setr_epi8(0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1));
        top0 = simde_mm_loadu_si128((simde__m128i *)(refSampMain+deltaInt));
        top1 = simde_mm_loadu_si128((simde__m128i *)(refSampMain+deltaInt+16));
        top0 = simde_mm_xor_si128(top0, simde_mm_setzero_si128()); // Redundant code. Visual Studio Express 2013 Release build messes up the order of operands in simde_mm_maddubs_epi16(), and hard to work around.
        top1 = simde_mm_xor_si128(top1, simde_mm_setzero_si128()); // Redundant code. Visual Studio Express 2013 Release build messes up the order of operands in simde_mm_maddubs_epi16(), and hard to work around.
        sum0 = simde_mm_add_epi16(simde_mm_set1_epi16(16), simde_mm_maddubs_epi16(top0, a0));
        sum1 = simde_mm_add_epi16(simde_mm_set1_epi16(16), simde_mm_maddubs_epi16(top1, a0));
        sum0 = simde_mm_srai_epi16(sum0, 5);
        sum1 = simde_mm_srai_epi16(sum1, 5);
        sum0 = simde_mm_packus_epi16(sum0, sum1);
        simde_mm_storeu_si128((simde__m128i *)p, sum0);
        p += 16;
      }
      p = tempBuf;
      for(colIndex=0; colIndex<2; colIndex++) {
        for(rowIndex=0; rowIndex<2; rowIndex++) {
          a0  = simde_mm_unpacklo_epi8(simde_mm_loadl_epi64((simde__m128i *)(p+0x00)), simde_mm_loadl_epi64((simde__m128i *)(p+0x10)));
          a1  = simde_mm_unpacklo_epi8(simde_mm_loadl_epi64((simde__m128i *)(p+0x20)), simde_mm_loadl_epi64((simde__m128i *)(p+0x30)));
          a2  = simde_mm_unpacklo_epi8(simde_mm_loadl_epi64((simde__m128i *)(p+0x40)), simde_mm_loadl_epi64((simde__m128i *)(p+0x50)));
          a3  = simde_mm_unpacklo_epi8(simde_mm_loadl_epi64((simde__m128i *)(p+0x60)), simde_mm_loadl_epi64((simde__m128i *)(p+0x70)));
          a4  = simde_mm_unpacklo_epi8(simde_mm_loadl_epi64((simde__m128i *)(p+0x80)), simde_mm_loadl_epi64((simde__m128i *)(p+0x90)));
          a5  = simde_mm_unpacklo_epi8(simde_mm_loadl_epi64((simde__m128i *)(p+0xA0)), simde_mm_loadl_epi64((simde__m128i *)(p+0xB0)));
          a6  = simde_mm_unpacklo_epi8(simde_mm_loadl_epi64((simde__m128i *)(p+0xC0)), simde_mm_loadl_epi64((simde__m128i *)(p+0xD0)));
          a7  = simde_mm_unpacklo_epi8(simde_mm_loadl_epi64((simde__m128i *)(p+0xE0)), simde_mm_loadl_epi64((simde__m128i *)(p+0xF0)));

          a8  = simde_mm_unpackhi_epi16(a0, a1);
          a0  = simde_mm_unpacklo_epi16(a0, a1);
          a9  = simde_mm_unpackhi_epi16(a2, a3);
          a2  = simde_mm_unpacklo_epi16(a2, a3);
          a10 = simde_mm_unpackhi_epi16(a4, a5);
          a4  = simde_mm_unpacklo_epi16(a4, a5);
          a11 = simde_mm_unpackhi_epi16(a6, a7);
          a6  = simde_mm_unpacklo_epi16(a6, a7);

          a1  = simde_mm_unpackhi_epi32(a0,  a2);
          a0  = simde_mm_unpacklo_epi32(a0,  a2);
          a3  = simde_mm_unpackhi_epi32(a4,  a6);
          a4  = simde_mm_unpacklo_epi32(a4,  a6);
          a5  = simde_mm_unpackhi_epi32(a8,  a9);
          a8  = simde_mm_unpacklo_epi32(a8,  a9);
          a7  = simde_mm_unpackhi_epi32(a10, a11);
          a10 = simde_mm_unpacklo_epi32(a10, a11);

          a2  = simde_mm_unpackhi_epi64(a0, a4);
          a0  = simde_mm_unpacklo_epi64(a0, a4);
          a6  = simde_mm_unpackhi_epi64(a8, a10);
          a8  = simde_mm_unpacklo_epi64(a8, a10);
          a9  = simde_mm_unpackhi_epi64(a1, a3);
          a1  = simde_mm_unpacklo_epi64(a1, a3);
          a11 = simde_mm_unpackhi_epi64(a5, a7);
          a5  = simde_mm_unpacklo_epi64(a5, a7);

          simde_mm_storeu_si128((simde__m128i *)predictionPtr, a0);  predictionPtr += predictionBufferStride;
          simde_mm_storeu_si128((simde__m128i *)predictionPtr, a2);  predictionPtr += predictionBufferStride;
          simde_mm_storeu_si128((simde__m128i *)predictionPtr, a1);  predictionPtr += predictionBufferStride;
          simde_mm_storeu_si128((simde__m128i *)predictionPtr, a9);  predictionPtr += predictionBufferStride;
          simde_mm_storeu_si128((simde__m128i *)predictionPtr, a8);  predictionPtr += predictionBufferStride;
          simde_mm_storeu_si128((simde__m128i *)predictionPtr, a6);  predictionPtr += predictionBufferStride;
          simde_mm_storeu_si128((simde__m128i *)predictionPtr, a5);  predictionPtr += predictionBufferStride;
          simde_mm_storeu_si128((simde__m128i *)predictionPtr, a11); predictionPtr += predictionBufferStride;
          p += 8;
        }
        p = tempBuf + 0x100;
        predictionPtr -= 16 * predictionBufferStride - 16;
      }
    }
  }
  else {
    if(size == 4) {
      for(colIndex=0; colIndex<size; colIndex+=2) {
        deltaSum += intraPredAngle;
        deltaInt   = deltaSum >> 5;
        deltaFract = deltaSum & 31;
        a0 = simde_mm_unpacklo_epi8(simde_mm_cvtsi32_si128(32 - deltaFract), simde_mm_cvtsi32_si128(deltaFract));
        top0 = simde_mm_loadl_epi64((simde__m128i *)(refSampMain+deltaInt));
        deltaSum += intraPredAngle;
        deltaInt   = deltaSum >> 5;
        deltaFract = deltaSum & 31;
        a1 = simde_mm_unpacklo_epi8(simde_mm_cvtsi32_si128(32 - deltaFract), simde_mm_cvtsi32_si128(deltaFract));
        a0 = simde_mm_unpacklo_epi16(a0, a1);
        a0 = simde_mm_shuffle_epi8(a0, simde_mm_setr_epi8(0, 1, 0, 1, 0, 1, 0, 1, 2, 3, 2, 3, 2, 3, 2, 3));
        top0 = simde_mm_castps_si128(simde_mm_loadh_pi(simde_mm_castsi128_ps(top0), (simde__m64 *)(refSampMain+deltaInt)));
        top0 = simde_mm_shuffle_epi8(top0, simde_mm_setr_epi8(0, 1, 1, 2, 2, 3, 3, 4, 8, 9, 9, 10, 10, 11, 11, 12));
        sum0 = simde_mm_add_epi16(simde_mm_set1_epi16(16), simde_mm_maddubs_epi16(top0, a0));
        sum0 = simde_mm_srai_epi16(sum0, 5);
        sum0 = simde_mm_packus_epi16(sum0, sum0);
        *(EB_U32 *)p = simde_mm_cvtsi128_si32(sum0);
        *(EB_U32 *)(p+4) = simde_mm_cvtsi128_si32(simde_mm_srli_si128(sum0, 4));
        p += 8;
      }
      a0 = simde_mm_loadu_si128((simde__m128i *)tempBuf);
      a0 = simde_mm_shuffle_epi8(a0, simde_mm_setr_epi8(0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15));
      *(EB_U32 *)predictionPtr = simde_mm_cvtsi128_si32(a0); a0 = simde_mm_srli_si128(a0, 4); predictionPtr += predictionBufferStride;
      *(EB_U32 *)predictionPtr = simde_mm_cvtsi128_si32(a0); a0 = simde_mm_srli_si128(a0, 4); predictionPtr += predictionBufferStride;
      *(EB_U32 *)predictionPtr = simde_mm_cvtsi128_si32(a0); a0 = simde_mm_srli_si128(a0, 4); predictionPtr += predictionBufferStride;
      *(EB_U32 *)predictionPtr = simde_mm_cvtsi128_si32(a0);
    }
    else if(size == 8) {
      for(colIndex=0; colIndex<size; colIndex++) {
        deltaSum += intraPredAngle;
        deltaInt   = deltaSum >> 5;
        deltaFract = deltaSum & 31;
        a0 = simde_mm_unpacklo_epi8(simde_mm_cvtsi32_si128(32 - deltaFract), simde_mm_cvtsi32_si128(deltaFract));
        a0 = simde_mm_shuffle_epi8(a0, simde_mm_setr_epi8(0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1));
        top0 = simde_mm_loadu_si128((simde__m128i *)(refSampMain+deltaInt));
        top0 = simde_mm_shuffle_epi8(top0, simde_mm_setr_epi8(0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8));
        sum0 = simde_mm_add_epi16(simde_mm_set1_epi16(16), simde_mm_maddubs_epi16(top0, a0));
        sum0 = simde_mm_srai_epi16(sum0, 5);
        sum0 = simde_mm_packus_epi16(sum0, sum0);
        simde_mm_storel_epi64((simde__m128i *)p, sum0);
        p += 8;
      }
      a0 = simde_mm_unpacklo_epi8(simde_mm_loadl_epi64((simde__m128i *)(tempBuf+0x00)), simde_mm_loadl_epi64((simde__m128i *)(tempBuf+0x08)));
      a1 = simde_mm_unpacklo_epi8(simde_mm_loadl_epi64((simde__m128i *)(tempBuf+0x10)), simde_mm_loadl_epi64((simde__m128i *)(tempBuf+0x18)));
      a2 = simde_mm_unpacklo_epi8(simde_mm_loadl_epi64((simde__m128i *)(tempBuf+0x20)), simde_mm_loadl_epi64((simde__m128i *)(tempBuf+0x28)));
      a3 = simde_mm_unpacklo_epi8(simde_mm_loadl_epi64((simde__m128i *)(tempBuf+0x30)), simde_mm_loadl_epi64((simde__m128i *)(tempBuf+0x38)));
      a4 = simde_mm_unpackhi_epi16(a0, a1);
      a0 = simde_mm_unpacklo_epi16(a0, a1);
      a5 = simde_mm_unpackhi_epi16(a2, a3);
      a2 = simde_mm_unpacklo_epi16(a2, a3);
      a1 = simde_mm_unpackhi_epi32(a0, a2);
      a0 = simde_mm_unpacklo_epi32(a0, a2);
      a3 = simde_mm_unpackhi_epi32(a4, a5);
      a4 = simde_mm_unpacklo_epi32(a4, a5);
      simde_mm_storel_epi64((simde__m128i *)predictionPtr, a0); predictionPtr += predictionBufferStride;
      simde_mm_storeh_epi64((simde__m128i *)predictionPtr, a0); predictionPtr += predictionBufferStride;
      simde_mm_storel_epi64((simde__m128i *)predictionPtr, a1); predictionPtr += predictionBufferStride;
      simde_mm_storeh_epi64((simde__m128i *)predictionPtr, a1); predictionPtr += predictionBufferStride;
      simde_mm_storel_epi64((simde__m128i *)predictionPtr, a4); predictionPtr += predictionBufferStride;
      simde_mm_storeh_epi64((simde__m128i *)predictionPtr, a4); predictionPtr += predictionBufferStride;
      simde_mm_storel_epi64((simde__m128i *)predictionPtr, a3); predictionPtr += predictionBufferStride;
      simde_mm_storeh_epi64((simde__m128i *)predictionPtr, a3);
    }
    else if(size == 16) {
      for(colIndex=0; colIndex<size; colIndex++) {
        deltaSum += intraPredAngle;
        deltaInt   = deltaSum >> 5;
        deltaFract = deltaSum & 31;
        a0 = simde_mm_unpacklo_epi8(simde_mm_cvtsi32_si128(32 - deltaFract), simde_mm_cvtsi32_si128(deltaFract));
        a0 = simde_mm_shuffle_epi8(a0, simde_mm_setr_epi8(0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1));
        top0 = simde_mm_loadu_si128((simde__m128i *)(refSampMain+deltaInt));
        top1 = simde_mm_loadu_si128((simde__m128i *)(refSampMain+deltaInt+1));
        top2 = simde_mm_unpacklo_epi8(top0, top1);
        top0 = simde_mm_unpackhi_epi8(top0, top1);
        sum0 = simde_mm_add_epi16(simde_mm_set1_epi16(16), simde_mm_maddubs_epi16(top2, a0));
        sum1 = simde_mm_add_epi16(simde_mm_set1_epi16(16), simde_mm_maddubs_epi16(top0, a0));
        sum0 = simde_mm_srai_epi16(sum0, 5);
        sum1 = simde_mm_srai_epi16(sum1, 5);
        sum0 = simde_mm_packus_epi16(sum0, sum1);
        simde_mm_storeu_si128((simde__m128i *)p, sum0);
        p += 16;
      }
      p = tempBuf;
      for(colIndex=0; colIndex<2; colIndex++) {
        a0  = simde_mm_unpacklo_epi8(simde_mm_loadl_epi64((simde__m128i *)(p+0x00)), simde_mm_loadl_epi64((simde__m128i *)(p+0x10)));
        a1  = simde_mm_unpacklo_epi8(simde_mm_loadl_epi64((simde__m128i *)(p+0x20)), simde_mm_loadl_epi64((simde__m128i *)(p+0x30)));
        a2  = simde_mm_unpacklo_epi8(simde_mm_loadl_epi64((simde__m128i *)(p+0x40)), simde_mm_loadl_epi64((simde__m128i *)(p+0x50)));
        a3  = simde_mm_unpacklo_epi8(simde_mm_loadl_epi64((simde__m128i *)(p+0x60)), simde_mm_loadl_epi64((simde__m128i *)(p+0x70)));
        a4  = simde_mm_unpacklo_epi8(simde_mm_loadl_epi64((simde__m128i *)(p+0x80)), simde_mm_loadl_epi64((simde__m128i *)(p+0x90)));
        a5  = simde_mm_unpacklo_epi8(simde_mm_loadl_epi64((simde__m128i *)(p+0xA0)), simde_mm_loadl_epi64((simde__m128i *)(p+0xB0)));
        a6  = simde_mm_unpacklo_epi8(simde_mm_loadl_epi64((simde__m128i *)(p+0xC0)), simde_mm_loadl_epi64((simde__m128i *)(p+0xD0)));
        a7  = simde_mm_unpacklo_epi8(simde_mm_loadl_epi64((simde__m128i *)(p+0xE0)), simde_mm_loadl_epi64((simde__m128i *)(p+0xF0)));

        a8  = simde_mm_unpackhi_epi16(a0, a1);
        a0  = simde_mm_unpacklo_epi16(a0, a1);
        a9  = simde_mm_unpackhi_epi16(a2, a3);
        a2  = simde_mm_unpacklo_epi16(a2, a3);
        a10 = simde_mm_unpackhi_epi16(a4, a5);
        a4  = simde_mm_unpacklo_epi16(a4, a5);
        a11 = simde_mm_unpackhi_epi16(a6, a7);
        a6  = simde_mm_unpacklo_epi16(a6, a7);

        a1  = simde_mm_unpackhi_epi32(a0,  a2);
        a0  = simde_mm_unpacklo_epi32(a0,  a2);
        a3  = simde_mm_unpackhi_epi32(a4,  a6);
        a4  = simde_mm_unpacklo_epi32(a4,  a6);
        a5  = simde_mm_unpackhi_epi32(a8,  a9);
        a8  = simde_mm_unpacklo_epi32(a8,  a9);
        a7  = simde_mm_unpackhi_epi32(a10, a11);
        a10 = simde_mm_unpacklo_epi32(a10, a11);

        a2  = simde_mm_unpackhi_epi64(a0, a4);
        a0  = simde_mm_unpacklo_epi64(a0, a4);
        a6  = simde_mm_unpackhi_epi64(a8, a10);
        a8  = simde_mm_unpacklo_epi64(a8, a10);
        a9  = simde_mm_unpackhi_epi64(a1, a3);
        a1  = simde_mm_unpacklo_epi64(a1, a3);
        a11 = simde_mm_unpackhi_epi64(a5, a7);
        a5  = simde_mm_unpacklo_epi64(a5, a7);

        simde_mm_storeu_si128((simde__m128i *)predictionPtr, a0);  predictionPtr += predictionBufferStride;
        simde_mm_storeu_si128((simde__m128i *)predictionPtr, a2);  predictionPtr += predictionBufferStride;
        simde_mm_storeu_si128((simde__m128i *)predictionPtr, a1);  predictionPtr += predictionBufferStride;
        simde_mm_storeu_si128((simde__m128i *)predictionPtr, a9);  predictionPtr += predictionBufferStride;
        simde_mm_storeu_si128((simde__m128i *)predictionPtr, a8);  predictionPtr += predictionBufferStride;
        simde_mm_storeu_si128((simde__m128i *)predictionPtr, a6);  predictionPtr += predictionBufferStride;
        simde_mm_storeu_si128((simde__m128i *)predictionPtr, a5);  predictionPtr += predictionBufferStride;
        simde_mm_storeu_si128((simde__m128i *)predictionPtr, a11); predictionPtr += predictionBufferStride;
        p += 8;
      }
    }
    else { // size == 32
      for(colIndex=0; colIndex<size; colIndex++) {
        deltaSum += intraPredAngle;
        deltaInt   = deltaSum >> 5;
        deltaFract = deltaSum & 31;
        a0 = simde_mm_unpacklo_epi8(simde_mm_cvtsi32_si128(32 - deltaFract), simde_mm_cvtsi32_si128(deltaFract));
        a0 = simde_mm_shuffle_epi8(a0, simde_mm_setr_epi8(0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1));
        aa0 = simde_mm256_setr_m128i(a0, a0);
        ttop0 = simde_mm256_loadu_si256((simde__m256i *)(refSampMain + deltaInt));
        ttop1 = simde_mm256_loadu_si256((simde__m256i *)(refSampMain + deltaInt + 1));
        ttop2 = simde_mm256_unpacklo_epi8(ttop0, ttop1);
        ttop0 = simde_mm256_unpackhi_epi8(ttop0, ttop1);
        ssum0 = simde_mm256_add_epi16(simde_mm256_set1_epi16(16), simde_mm256_maddubs_epi16(ttop2, aa0));
        ssum1 = simde_mm256_add_epi16(simde_mm256_set1_epi16(16), simde_mm256_maddubs_epi16(ttop0, aa0));
        ssum0 = simde_mm256_srai_epi16(ssum0, 5);
        ssum1 = simde_mm256_srai_epi16(ssum1, 5);
        ssum0 = simde_mm256_packus_epi16(ssum0, ssum1);
        simde_mm256_storeu_si256((simde__m256i *)p, ssum0);
        p += 32;
      }
      p = tempBuf;
      for(colIndex=0; colIndex<2; colIndex++) {
        for(rowIndex=0; rowIndex<4; rowIndex++) {
          a0  = simde_mm_unpacklo_epi8(simde_mm_loadl_epi64((simde__m128i *)(p+0x000)), simde_mm_loadl_epi64((simde__m128i *)(p+0x020)));
          a1  = simde_mm_unpacklo_epi8(simde_mm_loadl_epi64((simde__m128i *)(p+0x040)), simde_mm_loadl_epi64((simde__m128i *)(p+0x060)));
          a2  = simde_mm_unpacklo_epi8(simde_mm_loadl_epi64((simde__m128i *)(p+0x080)), simde_mm_loadl_epi64((simde__m128i *)(p+0x0A0)));
          a3  = simde_mm_unpacklo_epi8(simde_mm_loadl_epi64((simde__m128i *)(p+0x0C0)), simde_mm_loadl_epi64((simde__m128i *)(p+0x0E0)));
          a4  = simde_mm_unpacklo_epi8(simde_mm_loadl_epi64((simde__m128i *)(p+0x100)), simde_mm_loadl_epi64((simde__m128i *)(p+0x120)));
          a5  = simde_mm_unpacklo_epi8(simde_mm_loadl_epi64((simde__m128i *)(p+0x140)), simde_mm_loadl_epi64((simde__m128i *)(p+0x160)));
          a6  = simde_mm_unpacklo_epi8(simde_mm_loadl_epi64((simde__m128i *)(p+0x180)), simde_mm_loadl_epi64((simde__m128i *)(p+0x1A0)));
          a7  = simde_mm_unpacklo_epi8(simde_mm_loadl_epi64((simde__m128i *)(p+0x1C0)), simde_mm_loadl_epi64((simde__m128i *)(p+0x1E0)));

          a8  = simde_mm_unpackhi_epi16(a0, a1);
          a0  = simde_mm_unpacklo_epi16(a0, a1);
          a9  = simde_mm_unpackhi_epi16(a2, a3);
          a2  = simde_mm_unpacklo_epi16(a2, a3);
          a10 = simde_mm_unpackhi_epi16(a4, a5);
          a4  = simde_mm_unpacklo_epi16(a4, a5);
          a11 = simde_mm_unpackhi_epi16(a6, a7);
          a6  = simde_mm_unpacklo_epi16(a6, a7);

          a1  = simde_mm_unpackhi_epi32(a0,  a2);
          a0  = simde_mm_unpacklo_epi32(a0,  a2);
          a3  = simde_mm_unpackhi_epi32(a4,  a6);
          a4  = simde_mm_unpacklo_epi32(a4,  a6);
          a5  = simde_mm_unpackhi_epi32(a8,  a9);
          a8  = simde_mm_unpacklo_epi32(a8,  a9);
          a7  = simde_mm_unpackhi_epi32(a10, a11);
          a10 = simde_mm_unpacklo_epi32(a10, a11);

          a2  = simde_mm_unpackhi_epi64(a0, a4);
          a0  = simde_mm_unpacklo_epi64(a0, a4);
          a6  = simde_mm_unpackhi_epi64(a8, a10);
          a8  = simde_mm_unpacklo_epi64(a8, a10);
          a9  = simde_mm_unpackhi_epi64(a1, a3);
          a1  = simde_mm_unpacklo_epi64(a1, a3);
          a11 = simde_mm_unpackhi_epi64(a5, a7);
          a5  = simde_mm_unpacklo_epi64(a5, a7);

          simde_mm_storeu_si128((simde__m128i *)predictionPtr, a0);  predictionPtr += predictionBufferStride;
          simde_mm_storeu_si128((simde__m128i *)predictionPtr, a2);  predictionPtr += predictionBufferStride;
          simde_mm_storeu_si128((simde__m128i *)predictionPtr, a1);  predictionPtr += predictionBufferStride;
          simde_mm_storeu_si128((simde__m128i *)predictionPtr, a9);  predictionPtr += predictionBufferStride;
          simde_mm_storeu_si128((simde__m128i *)predictionPtr, a8);  predictionPtr += predictionBufferStride;
          simde_mm_storeu_si128((simde__m128i *)predictionPtr, a6);  predictionPtr += predictionBufferStride;
          simde_mm_storeu_si128((simde__m128i *)predictionPtr, a5);  predictionPtr += predictionBufferStride;
          simde_mm_storeu_si128((simde__m128i *)predictionPtr, a11); predictionPtr += predictionBufferStride;
          p += 8;
        }
        p = tempBuf + 0x200;
        predictionPtr -= 32 * predictionBufferStride - 16;
      }
    }
  }
}

/*  ************************************************************  Vertical IntraPred***********************
                                                                   NOTE: this function has been updated only when size == 32******************************************/
void IntraModeVerticalLuma_AVX2_INTRIN(
    const EB_U32      size,                   //input parameter, denotes the size of the current PU
    EB_U8            *refSamples,             //input parameter, pointer to the reference samples
    EB_U8            *predictionPtr,          //output parameter, pointer to the prediction
    const EB_U32      predictionBufferStride, //input parameter, denotes the stride for the prediction ptr
    const EB_BOOL     skip                    //skip one row 
    )
{
    EB_U32 topLeftOffset = size << 1;
    EB_U32 leftOffset = 0;
    EB_U32 topOffset = (size << 1 )+ 1;
    simde__m128i xmm0;
    EB_U32 pStride = predictionBufferStride;

    if (size != 32) {
        simde__m128i xmm_mask1, xmm_mask2, xmm_topLeft, xmm_topLeft_lo, xmm_topLeft_hi, xmm_mask_skip, xmm_top, xmm_left, xmm_left_lo, xmm_left_hi;

        xmm0 = simde_mm_setzero_si128();
        xmm_mask1 = simde_mm_slli_si128(simde_mm_set1_epi8((signed char)0xFF), 1);
        xmm_mask2 = simde_mm_srli_si128(xmm_mask1, 15);              
        
        xmm_topLeft = simde_mm_set_epi16((signed short)0xffff, (signed short)0xffff, (signed short)0xffff, (signed short)0xffff, (signed short)0xffff, (signed short)0xffff, (signed short)0xffff, *(EB_U16*)(refSamples + topLeftOffset));
        xmm_topLeft = simde_mm_unpacklo_epi8(xmm_topLeft, xmm0);         
        xmm_topLeft = simde_mm_unpacklo_epi16(xmm_topLeft, xmm_topLeft);        
        xmm_topLeft = simde_mm_unpacklo_epi32(xmm_topLeft, xmm_topLeft);        
        xmm_topLeft_hi = simde_mm_unpackhi_epi64(xmm_topLeft, xmm_topLeft);        
        xmm_topLeft_lo = simde_mm_unpacklo_epi64(xmm_topLeft, xmm_topLeft);        
       
        if (!skip) {

            if (size == 8) {
                            
                xmm_left = simde_mm_packus_epi16(simde_mm_add_epi16(simde_mm_srai_epi16(simde_mm_sub_epi16(simde_mm_unpacklo_epi8(simde_mm_loadl_epi64((simde__m128i *)(refSamples + leftOffset)), xmm0), xmm_topLeft_lo), 1), xmm_topLeft_hi), xmm0);      
                xmm_top = simde_mm_and_si128(simde_mm_loadl_epi64((simde__m128i *)(refSamples + topOffset)), xmm_mask1);         
                MACRO_VERTICAL_LUMA_8(xmm_left, xmm_mask2, xmm_top)
                predictionPtr += (pStride << 2);             
                MACRO_VERTICAL_LUMA_8(xmm_left, xmm_mask2, xmm_top)
            }
            else if (size == 16) { 
                xmm_left = simde_mm_loadu_si128((simde__m128i *)(refSamples + leftOffset));           
                xmm_left_lo = simde_mm_add_epi16(simde_mm_srai_epi16(simde_mm_sub_epi16(simde_mm_unpacklo_epi8(xmm_left, xmm0), xmm_topLeft_lo), 1), xmm_topLeft_hi);         
                xmm_left_hi = simde_mm_add_epi16(simde_mm_srai_epi16(simde_mm_sub_epi16(simde_mm_unpackhi_epi8(xmm_left, xmm0), xmm_topLeft_lo), 1), xmm_topLeft_hi);         
                xmm_left = simde_mm_packus_epi16(xmm_left_lo, xmm_left_hi);      
                xmm_top = simde_mm_and_si128(simde_mm_loadu_si128((simde__m128i *)(refSamples + topOffset)), xmm_mask1);         
                MACRO_VERTICAL_LUMA_16(xmm_left, xmm_mask2, xmm_top)
                predictionPtr += (pStride << 2);
                MACRO_VERTICAL_LUMA_16(xmm_left, xmm_mask2, xmm_top)
                predictionPtr += (pStride << 2);
                MACRO_VERTICAL_LUMA_16(xmm_left, xmm_mask2, xmm_top)
                predictionPtr += (pStride << 2);
                MACRO_VERTICAL_LUMA_16(xmm_left, xmm_mask2, xmm_top)
            }
            else {
                xmm_left = simde_mm_packus_epi16(simde_mm_add_epi16( simde_mm_srai_epi16(simde_mm_sub_epi16(simde_mm_unpacklo_epi8(simde_mm_cvtsi32_si128(*(EB_U32*)(refSamples + leftOffset)), xmm0), xmm_topLeft_lo), 1), xmm_topLeft_hi), xmm0);
                xmm_top = simde_mm_and_si128(simde_mm_loadl_epi64((simde__m128i *)(refSamples + topOffset)), xmm_mask1);         
                MACRO_VERTICAL_LUMA_4(xmm_left, xmm_mask2, xmm_top)
                predictionPtr += (pStride << 1);             
                MACRO_VERTICAL_LUMA_4(xmm_left, xmm_mask2, xmm_top)
            }
        }
        else{                                            
            pStride <<= 1;            
            xmm_mask_skip = simde_mm_set1_epi16(0x00FF);

            if (size == 8) {
                xmm_left = simde_mm_packus_epi16(simde_mm_add_epi16(simde_mm_srai_epi16(simde_mm_sub_epi16(simde_mm_and_si128(simde_mm_loadl_epi64((simde__m128i *)(refSamples + leftOffset)), xmm_mask_skip), xmm_topLeft_lo), 1), xmm_topLeft_hi), xmm0);
                xmm_top = simde_mm_and_si128(simde_mm_loadl_epi64((simde__m128i *)(refSamples + topOffset)), xmm_mask1);                            
                MACRO_VERTICAL_LUMA_8(xmm_left, xmm_mask2, xmm_top);
            }
            else if (size == 16) {
                xmm_left = simde_mm_packus_epi16(simde_mm_add_epi16(simde_mm_srai_epi16(simde_mm_sub_epi16(simde_mm_and_si128(simde_mm_loadu_si128((simde__m128i *)(refSamples + leftOffset)), xmm_mask_skip), xmm_topLeft_lo), 1), xmm_topLeft_hi), xmm0);
                xmm_top = simde_mm_and_si128(simde_mm_loadu_si128((simde__m128i *)(refSamples + topOffset)), xmm_mask1);                              
                MACRO_VERTICAL_LUMA_16(xmm_left, xmm_mask2, xmm_top)
                predictionPtr += (pStride << 2);                                  
                MACRO_VERTICAL_LUMA_16(xmm_left, xmm_mask2, xmm_top)
            }
            else {
                xmm_left = simde_mm_packus_epi16(simde_mm_add_epi16(simde_mm_srai_epi16(simde_mm_sub_epi16(simde_mm_and_si128(simde_mm_cvtsi32_si128(*(EB_U32*)(refSamples + leftOffset)), xmm_mask_skip), xmm_topLeft_lo), 1), xmm_topLeft_hi), xmm0);                 
                xmm_top = simde_mm_and_si128(simde_mm_cvtsi32_si128(*(EB_U32*)(refSamples + topOffset)), xmm_mask1);                     
                MACRO_VERTICAL_LUMA_4(xmm_left, xmm_mask2, xmm_top)
            }
        }
    }
    else {
        simde__m256i xmm0;
        EB_U64 size_to_write;
        EB_U32 count;

         // Each storeu calls stores 32 bytes. Hence each iteration stores 8 * 32 bytes.
        // Depending on skip, we need 4 or 2 iterations to store 32x32 bytes.
        size_to_write = 4 >> (skip ? 1 : 0);
        pStride = pStride << (skip ? 1 : 0);

        xmm0 = simde_mm256_loadu_si256((simde__m256i *)(refSamples + topOffset));
		
		for (count = 0; count < size_to_write; count ++) {
            simde_mm256_storeu_si256((simde__m256i *)(predictionPtr), xmm0);                    
            simde_mm256_storeu_si256((simde__m256i *)(predictionPtr + pStride), xmm0);          
            simde_mm256_storeu_si256((simde__m256i *)(predictionPtr + 2 * pStride), xmm0);      
            simde_mm256_storeu_si256((simde__m256i *)(predictionPtr + 3 * pStride), xmm0);         
                
            predictionPtr += (pStride << 2);                                          
            simde_mm256_storeu_si256((simde__m256i *)(predictionPtr), xmm0);                    
            simde_mm256_storeu_si256((simde__m256i *)(predictionPtr + pStride), xmm0);          
            simde_mm256_storeu_si256((simde__m256i *)(predictionPtr + 2 * pStride), xmm0);      
            simde_mm256_storeu_si256((simde__m256i *)(predictionPtr + 3 * pStride), xmm0);         
                
            predictionPtr += (pStride << 2);                                          
        }
    }

    return;
}

void IntraModeDCLuma_AVX2_INTRIN(
    const EB_U32      size,                       //input parameter, denotes the size of the current PU
    EB_U8            *refSamples,                 //input parameter, pointer to the reference samples
    EB_U8            *predictionPtr,              //output parameter, pointer to the prediction
    const EB_U32      predictionBufferStride,     //input parameter, denotes the stride for the prediction ptr
    const EB_BOOL     skip)                       //skip one row 
{
    simde__m128i xmm0 = simde_mm_setzero_si128();
    EB_U32 pStride = predictionBufferStride;
    EB_U32 topOffset = (size << 1) + 1;
    EB_U32 leftOffset = 0;

   if (size != 32) {

	    simde__m128i xmm_mask1 = simde_mm_slli_si128(simde_mm_set1_epi8((signed char)0xFF), 1);
        simde__m128i xmm_mask2 = simde_mm_srli_si128(xmm_mask1, 15);
        simde__m128i xmm_C2 = simde_mm_set1_epi16(0x0002);

        if (!skip) {

            if (size == 16) {
                simde__m128i xmm_predictionDcValue, xmm_top, xmm_left, xmm_sum, xmm_predictionPtr_0;
                simde__m128i xmm_top_lo, xmm_top_hi, xmm_left_lo, xmm_left_hi, xmm_predictionDcValue_16, xmm_predictionDcValue_16_x2, xmm_predictionDcValue_16_x3;
                xmm_top = simde_mm_loadu_si128((simde__m128i *)(refSamples + topOffset));
                xmm_left = simde_mm_loadu_si128((simde__m128i *)(refSamples + leftOffset));
                xmm_top_lo = simde_mm_unpacklo_epi8(xmm_top, xmm0);
                xmm_top_hi = simde_mm_unpackhi_epi8(xmm_top, xmm0);
                xmm_left_lo = simde_mm_unpacklo_epi8(xmm_left, xmm0);
                xmm_left_hi = simde_mm_unpackhi_epi8(xmm_left, xmm0);
                
                xmm_sum = simde_mm_add_epi32(simde_mm_sad_epu8(xmm_top, xmm0), simde_mm_sad_epu8(xmm_left, xmm0));                
                
                xmm_predictionDcValue = simde_mm_srli_epi32(simde_mm_add_epi32(simde_mm_add_epi32(simde_mm_srli_si128(xmm_sum, 8), xmm_sum), simde_mm_cvtsi32_si128(16)), 5);
                xmm_predictionDcValue = simde_mm_unpacklo_epi8(xmm_predictionDcValue, xmm_predictionDcValue);
                xmm_predictionDcValue = simde_mm_unpacklo_epi16(xmm_predictionDcValue, xmm_predictionDcValue);
                xmm_predictionDcValue = simde_mm_unpacklo_epi32(xmm_predictionDcValue, xmm_predictionDcValue);
                xmm_predictionDcValue = simde_mm_unpacklo_epi64(xmm_predictionDcValue, xmm_predictionDcValue);

                xmm_predictionDcValue_16 = simde_mm_srli_epi16(xmm_predictionDcValue, 8);
                xmm_predictionDcValue = simde_mm_and_si128(xmm_predictionDcValue, xmm_mask1);
                xmm_predictionDcValue_16_x2 = simde_mm_add_epi16(xmm_predictionDcValue_16, xmm_predictionDcValue_16);
                xmm_predictionDcValue_16_x3 = simde_mm_add_epi16(xmm_predictionDcValue_16_x2, xmm_predictionDcValue_16); 

                xmm_top = simde_mm_packus_epi16(simde_mm_srli_epi16(simde_mm_add_epi16(simde_mm_add_epi16(xmm_top_lo, xmm_predictionDcValue_16_x3), xmm_C2), 2), 
                                           simde_mm_srli_epi16(simde_mm_add_epi16(simde_mm_add_epi16(xmm_top_hi, xmm_predictionDcValue_16_x3), xmm_C2), 2));

                xmm_left = simde_mm_srli_si128(simde_mm_packus_epi16(simde_mm_srli_epi16(simde_mm_add_epi16(simde_mm_add_epi16(xmm_left_lo, xmm_predictionDcValue_16_x3), xmm_C2), 2), 
                                                           simde_mm_srli_epi16(simde_mm_add_epi16(simde_mm_add_epi16(xmm_left_hi, xmm_predictionDcValue_16_x3), xmm_C2), 2)), 1);

                xmm_predictionPtr_0 = simde_mm_or_si128(simde_mm_and_si128(simde_mm_srli_epi16(simde_mm_add_epi16(simde_mm_add_epi16(simde_mm_add_epi16(xmm_top_lo, xmm_left_lo), xmm_predictionDcValue_16_x2), xmm_C2), 2), xmm_mask2), simde_mm_and_si128(xmm_top, xmm_mask1));
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr), xmm_predictionPtr_0);

                simde_mm_storeu_si128((simde__m128i *)(predictionPtr+pStride), simde_mm_or_si128(simde_mm_and_si128(xmm_left, xmm_mask2), xmm_predictionDcValue));
                xmm_left = simde_mm_srli_si128(xmm_left, 1);
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr+2*pStride), simde_mm_or_si128(simde_mm_and_si128(xmm_left, xmm_mask2), xmm_predictionDcValue));
                xmm_left = simde_mm_srli_si128(xmm_left, 1);
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr+3*pStride), simde_mm_or_si128(simde_mm_and_si128(xmm_left, xmm_mask2), xmm_predictionDcValue));
                xmm_left = simde_mm_srli_si128(xmm_left, 1);

                predictionPtr += (pStride << 2);
                MACRO_VERTICAL_LUMA_16(xmm_left, xmm_mask2, xmm_predictionDcValue)
                predictionPtr += (pStride << 2);
                MACRO_VERTICAL_LUMA_16(xmm_left, xmm_mask2, xmm_predictionDcValue)
                predictionPtr += (pStride << 2);
                MACRO_VERTICAL_LUMA_16(xmm_left, xmm_mask2, xmm_predictionDcValue)
            }
            else if (size == 8) {

                simde__m128i xmm_left, xmm_top, xmm_top_lo, xmm_left_lo, xmm_predictionDcValue, xmm_predictionDcValue_16;
                simde__m128i xmm_predictionDcValue_16_x2, xmm_predictionDcValue_16_x3, xmm_predictionPtr_0;

                xmm_top = simde_mm_loadl_epi64((simde__m128i *)(refSamples + topOffset));
                xmm_left = simde_mm_loadl_epi64((simde__m128i *)(refSamples + leftOffset));

                xmm_top_lo = simde_mm_unpacklo_epi8(xmm_top, xmm0);
                xmm_left_lo = simde_mm_unpacklo_epi8(xmm_left, xmm0);

                xmm_predictionDcValue = simde_mm_srli_epi32(simde_mm_add_epi32(simde_mm_add_epi32(simde_mm_sad_epu8(xmm_top, xmm0), simde_mm_sad_epu8(xmm_left, xmm0)), simde_mm_cvtsi32_si128(8)), 4);
                xmm_predictionDcValue = simde_mm_unpacklo_epi8(xmm_predictionDcValue, xmm_predictionDcValue);
                xmm_predictionDcValue = simde_mm_unpacklo_epi16(xmm_predictionDcValue, xmm_predictionDcValue);
                xmm_predictionDcValue = simde_mm_unpacklo_epi32(xmm_predictionDcValue, xmm_predictionDcValue);
                xmm_predictionDcValue = simde_mm_unpacklo_epi64(xmm_predictionDcValue, xmm_predictionDcValue);
                
                xmm_predictionDcValue_16 = simde_mm_srli_epi16(xmm_predictionDcValue, 8);
                xmm_predictionDcValue = simde_mm_and_si128(xmm_predictionDcValue, xmm_mask1);
                
                xmm_predictionDcValue_16_x2 = simde_mm_add_epi16(xmm_predictionDcValue_16, xmm_predictionDcValue_16);
                xmm_predictionDcValue_16_x3 = simde_mm_add_epi16(xmm_predictionDcValue_16_x2, xmm_predictionDcValue_16);
                
                xmm_top  = simde_mm_srli_epi16(simde_mm_add_epi16(simde_mm_add_epi16(xmm_top_lo, xmm_predictionDcValue_16_x3), xmm_C2), 2);
                xmm_left = simde_mm_srli_epi16(simde_mm_add_epi16(simde_mm_add_epi16(xmm_left_lo, xmm_predictionDcValue_16_x3), xmm_C2), 2);                

                xmm_left = simde_mm_srli_si128(simde_mm_packus_epi16(xmm_left, xmm_left), 1);
                xmm_top  = simde_mm_and_si128(simde_mm_packus_epi16(xmm_top, xmm_top), xmm_mask1);

                xmm_predictionPtr_0 = simde_mm_or_si128(simde_mm_and_si128(simde_mm_srli_epi16(simde_mm_add_epi16(simde_mm_add_epi16(simde_mm_add_epi16(xmm_top_lo, xmm_left_lo), xmm_predictionDcValue_16_x2), xmm_C2), 2), xmm_mask2), xmm_top);
                simde_mm_storel_epi64((simde__m128i *)(predictionPtr), xmm_predictionPtr_0);
                
                simde_mm_storel_epi64((simde__m128i *)(predictionPtr+pStride), simde_mm_or_si128(simde_mm_and_si128(xmm_left, xmm_mask2), xmm_predictionDcValue));
                xmm_left = simde_mm_srli_si128(xmm_left, 1);
                simde_mm_storel_epi64((simde__m128i *)(predictionPtr + 2 * pStride), simde_mm_or_si128(simde_mm_and_si128(xmm_left, xmm_mask2), xmm_predictionDcValue));
                xmm_left = simde_mm_srli_si128(xmm_left, 1);
                simde_mm_storel_epi64((simde__m128i *)(predictionPtr + 3 * pStride), simde_mm_or_si128(simde_mm_and_si128(xmm_left, xmm_mask2), xmm_predictionDcValue));
                xmm_left = simde_mm_srli_si128(xmm_left, 1);

                predictionPtr += (pStride << 2);
                MACRO_VERTICAL_LUMA_8(xmm_left, xmm_mask2, xmm_predictionDcValue)
                
            } else {
                simde__m128i xmm_left, xmm_top, xmm_top_lo, xmm_left_lo, xmm_predictionDcValue, xmm_predictionDcValue_16;
                simde__m128i xmm_predictionDcValue_16_x2, xmm_predictionDcValue_16_x3, xmm_predictionPtr_0;

                xmm_top = simde_mm_cvtsi32_si128(*(EB_U32*)(refSamples + topOffset));
                xmm_left = simde_mm_cvtsi32_si128(*(EB_U32*)(refSamples + leftOffset));
    
                xmm_top_lo = simde_mm_unpacklo_epi8(xmm_top, xmm0);
                xmm_left_lo = simde_mm_unpacklo_epi8(xmm_left, xmm0);
                xmm_predictionDcValue = simde_mm_srli_epi32(simde_mm_add_epi32(simde_mm_add_epi32(simde_mm_sad_epu8(xmm_top, xmm0), simde_mm_sad_epu8(xmm_left, xmm0)), simde_mm_cvtsi32_si128(4)), 3);
                xmm_predictionDcValue = simde_mm_unpacklo_epi8(xmm_predictionDcValue, xmm_predictionDcValue);
                xmm_predictionDcValue = simde_mm_unpacklo_epi16(xmm_predictionDcValue, xmm_predictionDcValue);
                xmm_predictionDcValue = simde_mm_unpacklo_epi32(xmm_predictionDcValue, xmm_predictionDcValue);

                xmm_predictionDcValue_16 = simde_mm_srli_epi16(xmm_predictionDcValue, 8);
                xmm_predictionDcValue = simde_mm_and_si128(xmm_predictionDcValue, xmm_mask1);

                xmm_predictionDcValue_16_x2 = simde_mm_add_epi16(xmm_predictionDcValue_16, xmm_predictionDcValue_16);
                xmm_predictionDcValue_16_x3 = simde_mm_add_epi16(xmm_predictionDcValue_16_x2, xmm_predictionDcValue_16);
                
                xmm_top = simde_mm_srli_epi16(simde_mm_add_epi16(simde_mm_add_epi16(xmm_top_lo, xmm_predictionDcValue_16_x3), xmm_C2), 2);
                xmm_left = simde_mm_srli_epi16(simde_mm_add_epi16(simde_mm_add_epi16(xmm_left_lo, xmm_predictionDcValue_16_x3), xmm_C2), 2);
                xmm_top = simde_mm_and_si128(simde_mm_packus_epi16(xmm_top, xmm_top), xmm_mask1);
                xmm_left = simde_mm_srli_si128(simde_mm_packus_epi16(xmm_left, xmm_left), 1);
                
                xmm_predictionPtr_0 = simde_mm_or_si128(simde_mm_and_si128(simde_mm_srli_epi16(simde_mm_add_epi16(simde_mm_add_epi16(simde_mm_add_epi16(xmm_top_lo, xmm_left_lo), xmm_predictionDcValue_16_x2), xmm_C2), 2), xmm_mask2), xmm_top);
                *(EB_U32*)predictionPtr = simde_mm_cvtsi128_si32(xmm_predictionPtr_0);
               
                *(EB_U32*)(predictionPtr + pStride) = simde_mm_cvtsi128_si32(simde_mm_or_si128(simde_mm_and_si128(xmm_left, xmm_mask2), xmm_predictionDcValue));
                xmm_left = simde_mm_srli_si128(xmm_left, 1);

                *(EB_U32*)(predictionPtr + 2*pStride) = simde_mm_cvtsi128_si32(simde_mm_or_si128(simde_mm_and_si128(xmm_left, xmm_mask2), xmm_predictionDcValue));
                *(EB_U32*)(predictionPtr + 3*pStride) = simde_mm_cvtsi128_si32(simde_mm_or_si128(simde_mm_and_si128(simde_mm_srli_si128(xmm_left, 1), xmm_mask2), xmm_predictionDcValue));
            }
        }
        else {
            pStride <<= 1;

            simde__m128i xmm_skip_mask = simde_mm_set1_epi16(0x00FF);
            simde__m128i xmm_left, xmm_sum, xmm_top, xmm_top_lo, xmm_top_hi, xmm_left_skipped, xmm_predictionDcValue, xmm_predictionDcValue_16;
            simde__m128i xmm_predictionDcValue_16_x2, xmm_predictionDcValue_16_x3, xmm_predictionPtr_0;

            if (size == 16) {
                xmm_top = simde_mm_loadu_si128((simde__m128i *)(refSamples + topOffset));
                xmm_left = simde_mm_loadu_si128((simde__m128i *)(refSamples + leftOffset));

                xmm_top_lo = simde_mm_unpacklo_epi8(xmm_top, xmm0);
                xmm_top_hi = simde_mm_unpackhi_epi8(xmm_top, xmm0);
                
                xmm_left_skipped = simde_mm_and_si128(xmm_skip_mask, xmm_left);

                xmm_sum = simde_mm_add_epi32(simde_mm_sad_epu8(xmm_top, xmm0), simde_mm_sad_epu8(xmm_left, xmm0));

                xmm_predictionDcValue = simde_mm_srli_epi32(simde_mm_add_epi32(simde_mm_add_epi32(simde_mm_srli_si128(xmm_sum, 8), xmm_sum), simde_mm_cvtsi32_si128(16)), 5);
                xmm_predictionDcValue = simde_mm_unpacklo_epi8(xmm_predictionDcValue, xmm_predictionDcValue);
                xmm_predictionDcValue = simde_mm_unpacklo_epi16(xmm_predictionDcValue, xmm_predictionDcValue);
                xmm_predictionDcValue = simde_mm_unpacklo_epi32(xmm_predictionDcValue, xmm_predictionDcValue);
                xmm_predictionDcValue = simde_mm_unpacklo_epi64(xmm_predictionDcValue, xmm_predictionDcValue);
                
                xmm_predictionDcValue_16 = simde_mm_srli_epi16(xmm_predictionDcValue, 8);
                xmm_predictionDcValue = simde_mm_and_si128(xmm_predictionDcValue, xmm_mask1);
                
                xmm_predictionDcValue_16_x2 = simde_mm_add_epi16(xmm_predictionDcValue_16, xmm_predictionDcValue_16);
                xmm_predictionDcValue_16_x3 = simde_mm_add_epi16(xmm_predictionDcValue_16_x2, xmm_predictionDcValue_16);

                xmm_left = simde_mm_srli_epi16(simde_mm_add_epi16(simde_mm_add_epi16(xmm_left_skipped, xmm_predictionDcValue_16_x3), xmm_C2), 2);                
                xmm_left = simde_mm_srli_si128(simde_mm_packus_epi16(xmm_left, xmm_left), 1);
                
                xmm_top = simde_mm_and_si128(simde_mm_packus_epi16(simde_mm_srli_epi16(simde_mm_add_epi16(simde_mm_add_epi16(xmm_top_lo, xmm_predictionDcValue_16_x3), xmm_C2), 2), 
                                                         simde_mm_srli_epi16(simde_mm_add_epi16(simde_mm_add_epi16(xmm_top_hi, xmm_predictionDcValue_16_x3), xmm_C2), 2)), xmm_mask1);

                xmm_predictionPtr_0 = simde_mm_or_si128(simde_mm_and_si128(simde_mm_srli_epi16(simde_mm_add_epi16(simde_mm_add_epi16(simde_mm_add_epi16(xmm_top_lo, xmm_left_skipped), xmm_predictionDcValue_16_x2), xmm_C2), 2), xmm_mask2), xmm_top);
                simde_mm_storeu_si128((simde__m128i *)predictionPtr, xmm_predictionPtr_0);                
                
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + pStride), simde_mm_or_si128(simde_mm_and_si128(xmm_left, xmm_mask2), xmm_predictionDcValue));
                xmm_left = simde_mm_srli_si128(xmm_left, 1);

                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 2 * pStride), simde_mm_or_si128(simde_mm_and_si128(xmm_left, xmm_mask2), xmm_predictionDcValue));
                xmm_left = simde_mm_srli_si128(xmm_left, 1);

                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 3 * pStride), simde_mm_or_si128(simde_mm_and_si128(xmm_left, xmm_mask2), xmm_predictionDcValue));
                xmm_left = simde_mm_srli_si128(xmm_left, 1);
                predictionPtr += (pStride << 2);
                MACRO_VERTICAL_LUMA_16(xmm_left, xmm_mask2, xmm_predictionDcValue)
            }
            else {

                xmm_top = simde_mm_loadl_epi64((simde__m128i *)(refSamples + topOffset));
                xmm_left = simde_mm_loadl_epi64((simde__m128i *)(refSamples + leftOffset));

                xmm_top_lo = simde_mm_unpacklo_epi8(xmm_top, xmm0);
                xmm_left_skipped = simde_mm_and_si128(xmm_skip_mask, xmm_left);

                xmm_predictionDcValue = simde_mm_srli_epi32(simde_mm_add_epi32(simde_mm_add_epi32(simde_mm_sad_epu8(xmm_top, xmm0), simde_mm_sad_epu8(xmm_left, xmm0)), simde_mm_cvtsi32_si128(8)), 4);
                xmm_predictionDcValue = simde_mm_unpacklo_epi8(xmm_predictionDcValue, xmm_predictionDcValue);
                xmm_predictionDcValue = simde_mm_unpacklo_epi16(xmm_predictionDcValue, xmm_predictionDcValue);
                xmm_predictionDcValue = simde_mm_unpacklo_epi32(xmm_predictionDcValue, xmm_predictionDcValue);
                xmm_predictionDcValue = simde_mm_unpacklo_epi64(xmm_predictionDcValue, xmm_predictionDcValue);
                
                xmm_predictionDcValue_16 = simde_mm_srli_epi16(xmm_predictionDcValue, 8);
                xmm_predictionDcValue = simde_mm_and_si128(xmm_predictionDcValue, xmm_mask1);
                
                xmm_predictionDcValue_16_x2 = simde_mm_add_epi16(xmm_predictionDcValue_16, xmm_predictionDcValue_16);
                xmm_predictionDcValue_16_x3 = simde_mm_add_epi16(xmm_predictionDcValue_16_x2, xmm_predictionDcValue_16);

                xmm_top = simde_mm_srli_epi16(simde_mm_add_epi16(simde_mm_add_epi16(xmm_top_lo, xmm_predictionDcValue_16_x3), xmm_C2), 2);
                xmm_left = simde_mm_srli_epi16(simde_mm_add_epi16(simde_mm_add_epi16(xmm_left_skipped, xmm_predictionDcValue_16_x3), xmm_C2), 2);
                xmm_predictionPtr_0 = simde_mm_srli_epi16(simde_mm_add_epi16(simde_mm_add_epi16(simde_mm_add_epi16(xmm_top_lo, xmm_left_skipped), xmm_predictionDcValue_16_x2), xmm_C2), 2);

                xmm_left = simde_mm_srli_si128(simde_mm_packus_epi16(xmm_left, xmm_left), 1);
                xmm_predictionPtr_0 = simde_mm_or_si128(simde_mm_and_si128(xmm_predictionPtr_0, xmm_mask2), simde_mm_and_si128(simde_mm_packus_epi16(xmm_top, xmm_top), xmm_mask1));
                simde_mm_storel_epi64((simde__m128i *)predictionPtr, xmm_predictionPtr_0);
                
                simde_mm_storel_epi64((simde__m128i *)(predictionPtr + pStride), simde_mm_or_si128(simde_mm_and_si128(xmm_left, xmm_mask2), xmm_predictionDcValue));
                xmm_left = simde_mm_srli_si128(xmm_left, 1);
                
                simde_mm_storel_epi64((simde__m128i *)(predictionPtr + 2 * pStride), simde_mm_or_si128(simde_mm_and_si128(xmm_left, xmm_mask2), xmm_predictionDcValue));

                simde_mm_storel_epi64((simde__m128i *)(predictionPtr + 3 * pStride), simde_mm_or_si128(simde_mm_and_si128(simde_mm_srli_si128(xmm_left, 1), xmm_mask2), xmm_predictionDcValue));

            }
        }
    }
 
  

/*************************************************************************************************************************************************************************************************************/
  else {
	     simde__m256i xmm_sum,xmm_sadleft,xmm_sadtop,xmm_toptmp,xmm_lefttmp ,xmm_set,xmm_sum128_2,xmm_sum256,xmm_predictionDcValue;
		 simde__m256i xmm1 = simde_mm256_setzero_si256();
         simde__m128i xmm_sumhi,xmm_sumlo,xmm_sum1,xmm_sum128,xmm_sumhitmp,xmm_sumlotmp,xmm_movelotmp,xmm_movehitmp;
       
         xmm_sumhi = xmm_sumlo = xmm_sum128 = xmm_sumhitmp = xmm_sumlotmp = simde_mm_setzero_si128();
         xmm_sum = xmm_sadleft = xmm_sadtop =  xmm_toptmp = xmm_lefttmp  = simde_mm256_setzero_si256();
      
         xmm_toptmp  =simde_mm256_sad_epu8( simde_mm256_set_m128i ( simde_mm_loadu_si128( (simde__m128i *)(refSamples + topOffset +16) ),simde_mm_loadu_si128((simde__m128i *)(refSamples + topOffset))),xmm1);
         xmm_lefttmp =simde_mm256_sad_epu8( simde_mm256_set_m128i( simde_mm_loadu_si128((simde__m128i *)(refSamples + leftOffset+16)),simde_mm_loadu_si128((simde__m128i *)(refSamples + leftOffset))),xmm1);
        
	     xmm_sum = simde_mm256_add_epi32(xmm_toptmp, xmm_lefttmp  ) ;
	     xmm_sum = simde_mm256_hadd_epi32 (xmm_sum,xmm_sum);
		 xmm_sumlo =  simde_mm256_extracti128_si256(xmm_sum,0);
		 xmm_sumhi =  simde_mm256_extracti128_si256(xmm_sum,1);
      
	     xmm_movelotmp = simde_mm_move_epi64 (xmm_sumlo);
         xmm_movehitmp = simde_mm_move_epi64 (xmm_sumhi);
                    
         xmm_sum1 =   simde_mm_add_epi32(xmm_movelotmp,xmm_movehitmp);

         xmm_sum1 = simde_mm_hadd_epi32(xmm_sum1,xmm_sum1);

         xmm_sum256 = simde_mm256_castsi128_si256(xmm_sum1);


//#endif      
          xmm_set = simde_mm256_castsi128_si256(simde_mm_set1_epi32(32));
      
	      xmm_sum128_2 = simde_mm256_add_epi32(xmm_sum256, xmm_set); // add offset
          xmm_predictionDcValue = simde_mm256_srli_epi32(xmm_sum128_2,6); //simde_mm256_srli_epi32
       

          simde__m128i dc128      = simde_mm256_castsi256_si128(xmm_predictionDcValue); 
                
            EB_U8 dc         = simde_mm_cvtsi128_si32 (dc128);
           xmm_predictionDcValue = simde_mm256_set1_epi8(dc);//simde_mm_broadcastb_epi8


            EB_U32 count;

            for (count = 0; count < 2; ++count) {

              simde_mm256_storeu_si256((simde__m256i *) predictionPtr, xmm_predictionDcValue);         
              simde_mm256_storeu_si256((simde__m256i *)(predictionPtr + 1 * pStride), xmm_predictionDcValue);
              simde_mm256_storeu_si256((simde__m256i *)(predictionPtr + 2 * pStride), xmm_predictionDcValue);
              simde_mm256_storeu_si256((simde__m256i *)(predictionPtr + 3 * pStride), xmm_predictionDcValue);
          
              predictionPtr += (pStride << 2);

             simde_mm256_storeu_si256((simde__m256i *) predictionPtr, xmm_predictionDcValue);  
             simde_mm256_storeu_si256((simde__m256i *)(predictionPtr + 1 * pStride), xmm_predictionDcValue);
             simde_mm256_storeu_si256((simde__m256i *)(predictionPtr + 2 * pStride), xmm_predictionDcValue);
             simde_mm256_storeu_si256((simde__m256i *)(predictionPtr + 3 * pStride), xmm_predictionDcValue);
          
              predictionPtr += (pStride << 2);

              simde_mm256_storeu_si256((simde__m256i *) predictionPtr, xmm_predictionDcValue);         
              simde_mm256_storeu_si256((simde__m256i *)(predictionPtr + 1 * pStride), xmm_predictionDcValue);
              simde_mm256_storeu_si256((simde__m256i *)(predictionPtr + 2 * pStride), xmm_predictionDcValue);
              simde_mm256_storeu_si256((simde__m256i *)(predictionPtr + 3 * pStride), xmm_predictionDcValue);
          
              predictionPtr += (pStride << 2);

             simde_mm256_storeu_si256((simde__m256i *) predictionPtr, xmm_predictionDcValue);  
             simde_mm256_storeu_si256((simde__m256i *)(predictionPtr + 1 * pStride), xmm_predictionDcValue);
             simde_mm256_storeu_si256((simde__m256i *)(predictionPtr + 2 * pStride), xmm_predictionDcValue);
             simde_mm256_storeu_si256((simde__m256i *)(predictionPtr + 3 * pStride), xmm_predictionDcValue);

              predictionPtr += (pStride << 2);

           
          }


     }


}

/***************************************************************************************************************************************************************************/
/***************************************************************************************IntraModeAngular_2_AVX2_INTRIN***************************************************************************************/
void IntraModeAngular_2_AVX2_INTRIN(
    const EB_U32      size,                       //input parameter, denotes the size of the current PU
    EB_U8            *refSamples,                 //input parameter, pointer to the reference samples
    EB_U8            *predictionPtr,              //output parameter, pointer to the prediction
    const EB_U32      predictionBufferStride,     //input parameter, denotes the stride for the prediction ptr
    const EB_BOOL     skip)                       //skip one row 
{
    EB_U32 pStride = predictionBufferStride;
    EB_U32 leftOffset = 0;

    if (!skip) {

        if (size == 32) {
            EB_U32 count; 
            for (count = 0; count < 8; ++count){
                simde_mm256_storeu_si256((simde__m256i *)predictionPtr,                      simde_mm256_loadu_si256((simde__m256i *)(refSamples + leftOffset + 1)));
                simde_mm256_storeu_si256((simde__m256i *)(predictionPtr + pStride),          simde_mm256_loadu_si256((simde__m256i *)(refSamples + leftOffset + 2)));
                simde_mm256_storeu_si256((simde__m256i *)(predictionPtr + 2 * pStride),      simde_mm256_loadu_si256((simde__m256i *)(refSamples + leftOffset + 3)));
                simde_mm256_storeu_si256((simde__m256i *)(predictionPtr + 3 * pStride),      simde_mm256_loadu_si256((simde__m256i *)(refSamples + leftOffset + 4)));
               
                refSamples += 4;
                predictionPtr += (pStride << 2);
            }
        }
        else if (size == 16) {
            simde_mm_storeu_si128((simde__m128i *)predictionPtr,                 simde_mm_loadu_si128((simde__m128i *)(refSamples + leftOffset + 1)));
            simde_mm_storeu_si128((simde__m128i *)(predictionPtr + pStride),     simde_mm_loadu_si128((simde__m128i *)(refSamples + leftOffset + 2)));
            simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 2 * pStride), simde_mm_loadu_si128((simde__m128i *)(refSamples + leftOffset + 3)));
            simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 3 * pStride), simde_mm_loadu_si128((simde__m128i *)(refSamples + leftOffset + 4)));
            predictionPtr += (pStride << 2);
            simde_mm_storeu_si128((simde__m128i *)predictionPtr,                 simde_mm_loadu_si128((simde__m128i *)(refSamples + leftOffset + 5)));
            simde_mm_storeu_si128((simde__m128i *)(predictionPtr + pStride),     simde_mm_loadu_si128((simde__m128i *)(refSamples + leftOffset + 6)));
            simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 2 * pStride), simde_mm_loadu_si128((simde__m128i *)(refSamples + leftOffset + 7)));
            simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 3 * pStride), simde_mm_loadu_si128((simde__m128i *)(refSamples + leftOffset + 8)));
            predictionPtr += (pStride << 2);
            simde_mm_storeu_si128((simde__m128i *)predictionPtr,                 simde_mm_loadu_si128((simde__m128i *)(refSamples + leftOffset + 9)));
            simde_mm_storeu_si128((simde__m128i *)(predictionPtr + pStride),     simde_mm_loadu_si128((simde__m128i *)(refSamples + leftOffset + 10)));
            simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 2 * pStride), simde_mm_loadu_si128((simde__m128i *)(refSamples + leftOffset + 11)));
            simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 3 * pStride), simde_mm_loadu_si128((simde__m128i *)(refSamples + leftOffset + 12)));
            predictionPtr += (pStride << 2);
            simde_mm_storeu_si128((simde__m128i *)predictionPtr,                 simde_mm_loadu_si128((simde__m128i *)(refSamples + leftOffset + 13)));
            simde_mm_storeu_si128((simde__m128i *)(predictionPtr + pStride),     simde_mm_loadu_si128((simde__m128i *)(refSamples + leftOffset + 14)));
            simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 2 * pStride), simde_mm_loadu_si128((simde__m128i *)(refSamples + leftOffset + 15)));
            simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 3 * pStride), simde_mm_loadu_si128((simde__m128i *)(refSamples + leftOffset + 16)));
        }
        else if (size == 8) {
            simde_mm_storel_epi64((simde__m128i *)predictionPtr,                 simde_mm_loadl_epi64((simde__m128i *)(refSamples + leftOffset + 1)));
            simde_mm_storel_epi64((simde__m128i *)(predictionPtr + pStride),     simde_mm_loadl_epi64((simde__m128i *)(refSamples + leftOffset + 2)));
            simde_mm_storel_epi64((simde__m128i *)(predictionPtr + 2 * pStride), simde_mm_loadl_epi64((simde__m128i *)(refSamples + leftOffset + 3)));
            simde_mm_storel_epi64((simde__m128i *)(predictionPtr + 3 * pStride), simde_mm_loadl_epi64((simde__m128i *)(refSamples + leftOffset + 4)));
            predictionPtr += (pStride << 2);
            simde_mm_storel_epi64((simde__m128i *)predictionPtr,                 simde_mm_loadl_epi64((simde__m128i *)(refSamples + leftOffset + 5)));
            simde_mm_storel_epi64((simde__m128i *)(predictionPtr + pStride),     simde_mm_loadl_epi64((simde__m128i *)(refSamples + leftOffset + 6)));
            simde_mm_storel_epi64((simde__m128i *)(predictionPtr + 2 * pStride), simde_mm_loadl_epi64((simde__m128i *)(refSamples + leftOffset + 7)));
            simde_mm_storel_epi64((simde__m128i *)(predictionPtr + 3 * pStride), simde_mm_loadl_epi64((simde__m128i *)(refSamples + leftOffset + 8)));
        }
        else {
            *(EB_U32 *)predictionPtr = *(EB_U32 *)(refSamples + leftOffset + 1);
            *(EB_U32 *)(predictionPtr + pStride) = *(EB_U32 *)(refSamples + leftOffset + 2);
            *(EB_U32 *)(predictionPtr + 2 * pStride) = *(EB_U32 *)(refSamples + leftOffset + 3);
            *(EB_U32 *)(predictionPtr + 3 * pStride) = *(EB_U32 *)(refSamples + leftOffset + 4);
        }
    }
    else {
        if (size != 4) {
            pStride <<= 1;

            if (size == 32) {
                EB_U32 count;

                for (count = 0; count < 4; ++count) {
                    
                    simde_mm256_storeu_si256((simde__m256i *)predictionPtr,                      simde_mm256_loadu_si256((simde__m256i *)(refSamples + leftOffset + 1)));
                    simde_mm256_storeu_si256((simde__m256i *)(predictionPtr + pStride),          simde_mm256_loadu_si256((simde__m256i *)(refSamples + leftOffset + 3)));
                    simde_mm256_storeu_si256((simde__m256i *)(predictionPtr + 2 * pStride),      simde_mm256_loadu_si256((simde__m256i *)(refSamples + leftOffset + 5)));
                    simde_mm256_storeu_si256((simde__m256i *)(predictionPtr + 3 * pStride),      simde_mm256_loadu_si256((simde__m256i *)(refSamples + leftOffset + 7)));
                    refSamples += 8;
                    predictionPtr += (pStride << 2);
                }
            }
            else if (size == 16) {
                
                simde_mm_storeu_si128((simde__m128i *)predictionPtr,                 simde_mm_loadu_si128((simde__m128i *)(refSamples + leftOffset + 1)));
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + pStride),     simde_mm_loadu_si128((simde__m128i *)(refSamples + leftOffset + 3)));
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 2 * pStride), simde_mm_loadu_si128((simde__m128i *)(refSamples + leftOffset + 5)));
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 3 * pStride), simde_mm_loadu_si128((simde__m128i *)(refSamples + leftOffset + 7)));
                predictionPtr += (pStride << 2);
                simde_mm_storeu_si128((simde__m128i *)predictionPtr,                 simde_mm_loadu_si128((simde__m128i *)(refSamples + leftOffset + 9)));
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + pStride),     simde_mm_loadu_si128((simde__m128i *)(refSamples + leftOffset + 11)));
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 2 * pStride), simde_mm_loadu_si128((simde__m128i *)(refSamples + leftOffset + 13)));
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 3 * pStride), simde_mm_loadu_si128((simde__m128i *)(refSamples + leftOffset + 15)));
            }
            else {
                simde_mm_storel_epi64((simde__m128i *)predictionPtr,                 simde_mm_loadl_epi64((simde__m128i *)(refSamples + leftOffset + 1)));
                simde_mm_storel_epi64((simde__m128i *)(predictionPtr + pStride),     simde_mm_loadl_epi64((simde__m128i *)(refSamples + leftOffset + 3)));
                simde_mm_storel_epi64((simde__m128i *)(predictionPtr + 2 * pStride), simde_mm_loadl_epi64((simde__m128i *)(refSamples + leftOffset + 5)));
                simde_mm_storel_epi64((simde__m128i *)(predictionPtr + 3 * pStride), simde_mm_loadl_epi64((simde__m128i *)(refSamples + leftOffset + 7)));                
            }
        }
        else {
            *(EB_U32*)predictionPtr = *(EB_U32*)(refSamples + leftOffset + 1);
            *(EB_U32*)(predictionPtr + 2 * pStride) = *(EB_U32*)(refSamples + leftOffset + 3);
        }
    }
}

#define MIDRANGE_VALUE_8BIT    128

EB_U32 UpdateNeighborDcIntraPred_AVX2_INTRIN(
	EB_U8                           *yIntraReferenceArrayReverse,
    EB_U16                           inputHeight,
    EB_U16                           strideY,
    EB_BYTE                          bufferY,
    EB_U16                           originY,
    EB_U16                           originX,
	EB_U32                           srcOriginX,
	EB_U32                           srcOriginY,
	EB_U32                           blockSize)
{

	EB_U32 idx;
	EB_U8  *srcPtr;
	EB_U8  *dstPtr;
	EB_U8  *readPtr;

	//EB_U32 count;
        (void) inputHeight;
	EB_U8 *yBorderReverse = yIntraReferenceArrayReverse;
	//EB_U32 height = inputHeight;
	//EB_U32 blockSizeHalf = blockSize << 1;
	EB_U32 topOffset = (blockSize << 1) + 1;
	EB_U32 leftOffset = 0;
	EB_U32	stride = strideY;
	simde__m128i xmm0 = simde_mm_setzero_si128();
	simde__m256i xmm1 = simde_mm256_setzero_si256();
	simde__m256i ymm0;

	simde__m128i xmm_sad = simde_mm_setzero_si128();

	// Adjust the Source ptr to start at the origin of the block being updated
	srcPtr = bufferY + (((srcOriginY + originY) * stride) + (srcOriginX + originX));

	// Adjust the Destination ptr to start at the origin of the Intra reference array
	dstPtr = yBorderReverse;

//CHKn here we need ref on Top+Left only. and memset is done only for border CUs

	//Initialise the Luma Intra Reference Array to the mid range value 128 (for CUs at the picture boundaries)
	memset(dstPtr, MIDRANGE_VALUE_8BIT, (blockSize << 2) + 1);

	// Get the left-column
	//count = blockSizeHalf;

	readPtr = srcPtr - 1;


	if (blockSize != 32) {

		simde__m128i xmm_mask1 = simde_mm_slli_si128(simde_mm_set1_epi8((signed char)0xFF), 1);
		simde__m128i xmm_mask2 = simde_mm_srli_si128(xmm_mask1, 15);
		simde__m128i xmm_C2 = simde_mm_set1_epi16(0x0002);


		if (blockSize == 16) {
			simde__m128i xmm_predictionDcValue, xmm_top, xmm_left, xmm_sum, xmm_predictionPtr_0;
			simde__m128i xmm_top_lo, xmm_top_hi, xmm_left_lo, xmm_left_hi, xmm_predictionDcValue_16, xmm_predictionDcValue_16_x2, xmm_predictionDcValue_16_x3;
			if (srcOriginY != 0)
			{
				xmm_top = simde_mm_loadu_si128((simde__m128i *)(srcPtr - stride));
			}
			else
			{
				xmm_top = simde_mm_loadu_si128((simde__m128i *)(yBorderReverse + topOffset));//simde_mm_set1_epi8(128);
			}

			if (srcOriginX != 0) {
				xmm_left = simde_mm_set_epi8(*(readPtr + 15 * stride), *(readPtr + 14 * stride), *(readPtr + 13 * stride), *(readPtr + 12 * stride), *(readPtr + 11 * stride), *(readPtr + 10 * stride), *(readPtr + 9 * stride), *(readPtr + 8 * stride), *(readPtr + 7 * stride), *(readPtr + 6 * stride), *(readPtr + 5 * stride), *(readPtr + 4 * stride), *(readPtr + 3 * stride), *(readPtr + 2 * stride), *(readPtr + stride), *readPtr); //simde_mm_loadu_si128((simde__m128i *)(yBorderReverse + leftOffset));
			}
			else
			{
				xmm_left = simde_mm_loadu_si128((simde__m128i *)(yBorderReverse + leftOffset));
			}

			xmm_top_lo = simde_mm_unpacklo_epi8(xmm_top, xmm0);
			xmm_top_hi = simde_mm_unpackhi_epi8(xmm_top, xmm0);
			xmm_left_lo = simde_mm_unpacklo_epi8(xmm_left, xmm0);
			xmm_left_hi = simde_mm_unpackhi_epi8(xmm_left, xmm0);

			xmm_sum = simde_mm_add_epi32(simde_mm_sad_epu8(xmm_top, xmm0), simde_mm_sad_epu8(xmm_left, xmm0));

			xmm_predictionDcValue = simde_mm_srli_epi32(simde_mm_add_epi32(simde_mm_add_epi32(simde_mm_srli_si128(xmm_sum, 8), xmm_sum), simde_mm_cvtsi32_si128(16)), 5);
			xmm_predictionDcValue = simde_mm_unpacklo_epi8(xmm_predictionDcValue, xmm_predictionDcValue);
			xmm_predictionDcValue = simde_mm_unpacklo_epi16(xmm_predictionDcValue, xmm_predictionDcValue);
			xmm_predictionDcValue = simde_mm_unpacklo_epi32(xmm_predictionDcValue, xmm_predictionDcValue);
			xmm_predictionDcValue = simde_mm_unpacklo_epi64(xmm_predictionDcValue, xmm_predictionDcValue);

			xmm_predictionDcValue_16 = simde_mm_srli_epi16(xmm_predictionDcValue, 8);
			xmm_predictionDcValue = simde_mm_and_si128(xmm_predictionDcValue, xmm_mask1);
			xmm_predictionDcValue_16_x2 = simde_mm_add_epi16(xmm_predictionDcValue_16, xmm_predictionDcValue_16);
			xmm_predictionDcValue_16_x3 = simde_mm_add_epi16(xmm_predictionDcValue_16_x2, xmm_predictionDcValue_16);

			xmm_top = simde_mm_packus_epi16(simde_mm_srli_epi16(simde_mm_add_epi16(simde_mm_add_epi16(xmm_top_lo, xmm_predictionDcValue_16_x3), xmm_C2), 2),
				simde_mm_srli_epi16(simde_mm_add_epi16(simde_mm_add_epi16(xmm_top_hi, xmm_predictionDcValue_16_x3), xmm_C2), 2));

			xmm_left = simde_mm_srli_si128(simde_mm_packus_epi16(simde_mm_srli_epi16(simde_mm_add_epi16(simde_mm_add_epi16(xmm_left_lo, xmm_predictionDcValue_16_x3), xmm_C2), 2),
				simde_mm_srli_epi16(simde_mm_add_epi16(simde_mm_add_epi16(xmm_left_hi, xmm_predictionDcValue_16_x3), xmm_C2), 2)), 1);

			xmm_predictionPtr_0 = simde_mm_or_si128(simde_mm_and_si128(simde_mm_srli_epi16(simde_mm_add_epi16(simde_mm_add_epi16(simde_mm_add_epi16(xmm_top_lo, xmm_left_lo), xmm_predictionDcValue_16_x2), xmm_C2), 2), xmm_mask2), simde_mm_and_si128(xmm_top, xmm_mask1));


			xmm_sad = simde_mm_add_epi32(xmm_sad, simde_mm_sad_epu8(simde_mm_loadu_si128((simde__m128i*)(srcPtr)), xmm_predictionPtr_0));
			xmm_sad = simde_mm_add_epi32(xmm_sad, simde_mm_sad_epu8(simde_mm_loadu_si128((simde__m128i*)(srcPtr + stride)), simde_mm_or_si128(simde_mm_and_si128(xmm_left, xmm_mask2), xmm_predictionDcValue)));
			xmm_left = simde_mm_srli_si128(xmm_left, 1);
			xmm_sad = simde_mm_add_epi32(xmm_sad, simde_mm_sad_epu8(simde_mm_loadu_si128((simde__m128i*)(srcPtr + (stride << 1))), simde_mm_or_si128(simde_mm_and_si128(xmm_left, xmm_mask2), xmm_predictionDcValue)));
			xmm_left = simde_mm_srli_si128(xmm_left, 1);
			xmm_sad = simde_mm_add_epi32(xmm_sad, simde_mm_sad_epu8(simde_mm_loadu_si128((simde__m128i*)(srcPtr + 3 * stride)), simde_mm_or_si128(simde_mm_and_si128(xmm_left, xmm_mask2), xmm_predictionDcValue)));
			xmm_left = simde_mm_srli_si128(xmm_left, 1);

			srcPtr += (stride << 2);

			for (idx = 4; idx < blockSize; idx += 4){
				xmm_sad = simde_mm_add_epi32(xmm_sad, simde_mm_sad_epu8(simde_mm_loadu_si128((simde__m128i*)(srcPtr)), simde_mm_or_si128(simde_mm_and_si128(xmm_left, xmm_mask2), xmm_predictionDcValue)));
				xmm_left = simde_mm_srli_si128(xmm_left, 1);
				xmm_sad = simde_mm_add_epi32(xmm_sad, simde_mm_sad_epu8(simde_mm_loadu_si128((simde__m128i*)(srcPtr + stride)), simde_mm_or_si128(simde_mm_and_si128(xmm_left, xmm_mask2), xmm_predictionDcValue)));
				xmm_left = simde_mm_srli_si128(xmm_left, 1);
				xmm_sad = simde_mm_add_epi32(xmm_sad, simde_mm_sad_epu8(simde_mm_loadu_si128((simde__m128i*)(srcPtr + (stride << 1))), simde_mm_or_si128(simde_mm_and_si128(xmm_left, xmm_mask2), xmm_predictionDcValue)));
				xmm_left = simde_mm_srli_si128(xmm_left, 1);
				xmm_sad = simde_mm_add_epi32(xmm_sad, simde_mm_sad_epu8(simde_mm_loadu_si128((simde__m128i*)(srcPtr + 3 * stride)), simde_mm_or_si128(simde_mm_and_si128(xmm_left, xmm_mask2), xmm_predictionDcValue)));
				xmm_left = simde_mm_srli_si128(xmm_left, 1);

				srcPtr += (stride << 2);

			}

			xmm_sad = simde_mm_add_epi32(xmm_sad, simde_mm_srli_si128(xmm_sad, 8));
			return simde_mm_cvtsi128_si32(xmm_sad);
		}

        else {


			simde__m128i xmm_left, xmm_top, xmm_top_lo, xmm_left_lo, xmm_predictionDcValue, xmm_predictionDcValue_16;
			simde__m128i xmm_predictionDcValue_16_x2, xmm_predictionDcValue_16_x3, xmm_predictionPtr_0;
			if (srcOriginY != 0)
			{
				xmm_top = simde_mm_loadl_epi64((simde__m128i *)(srcPtr - stride));
			}
			else
			{
				xmm_top = simde_mm_loadl_epi64((simde__m128i *)(yBorderReverse + topOffset));//simde_mm_set1_epi8(128);//
			}

			if (srcOriginX != 0) {
				xmm_left = simde_mm_set_epi8(0, 0, 0, 0, 0, 0, 0, 0, *(readPtr + 7 * stride), *(readPtr + 6 * stride), *(readPtr + 5 * stride), *(readPtr + 4 * stride), *(readPtr + 3 * stride), *(readPtr + 2 * stride), *(readPtr + stride), *readPtr); //simde_mm_loadu_si128((simde__m128i *)(yBorderReverse + leftOffset));
			}
			else
			{
				xmm_left = simde_mm_loadl_epi64((simde__m128i *)(yBorderReverse + leftOffset));
			}

			xmm_top_lo = simde_mm_unpacklo_epi8(xmm_top, xmm0);
			xmm_left_lo = simde_mm_unpacklo_epi8(xmm_left, xmm0);

			xmm_predictionDcValue = simde_mm_srli_epi32(simde_mm_add_epi32(simde_mm_add_epi32(simde_mm_sad_epu8(xmm_top, xmm0), simde_mm_sad_epu8(xmm_left, xmm0)), simde_mm_cvtsi32_si128(8)), 4);
			xmm_predictionDcValue = simde_mm_unpacklo_epi8(xmm_predictionDcValue, xmm_predictionDcValue);
			xmm_predictionDcValue = simde_mm_unpacklo_epi16(xmm_predictionDcValue, xmm_predictionDcValue);
			xmm_predictionDcValue = simde_mm_unpacklo_epi32(xmm_predictionDcValue, xmm_predictionDcValue);
			xmm_predictionDcValue = simde_mm_unpacklo_epi64(xmm_predictionDcValue, xmm_predictionDcValue);

			xmm_predictionDcValue_16 = simde_mm_srli_epi16(xmm_predictionDcValue, 8);
			xmm_predictionDcValue = simde_mm_and_si128(xmm_predictionDcValue, xmm_mask1);

			xmm_predictionDcValue_16_x2 = simde_mm_add_epi16(xmm_predictionDcValue_16, xmm_predictionDcValue_16);
			xmm_predictionDcValue_16_x3 = simde_mm_add_epi16(xmm_predictionDcValue_16_x2, xmm_predictionDcValue_16);

			xmm_top = simde_mm_srli_epi16(simde_mm_add_epi16(simde_mm_add_epi16(xmm_top_lo, xmm_predictionDcValue_16_x3), xmm_C2), 2);
			xmm_left = simde_mm_srli_epi16(simde_mm_add_epi16(simde_mm_add_epi16(xmm_left_lo, xmm_predictionDcValue_16_x3), xmm_C2), 2);

			xmm_left = simde_mm_srli_si128(simde_mm_packus_epi16(xmm_left, xmm_left), 1);
			xmm_top = simde_mm_and_si128(simde_mm_packus_epi16(xmm_top, xmm_top), xmm_mask1);

			xmm_predictionPtr_0 = simde_mm_or_si128(simde_mm_and_si128(simde_mm_srli_epi16(simde_mm_add_epi16(simde_mm_add_epi16(simde_mm_add_epi16(xmm_top_lo, xmm_left_lo), xmm_predictionDcValue_16_x2), xmm_C2), 2), xmm_mask2), xmm_top);



			xmm_sad = simde_mm_setzero_si128();

			xmm_sad = simde_mm_add_epi32(xmm_sad, simde_mm_sad_epu8(simde_mm_loadu_si128((simde__m128i*)(srcPtr)), xmm_predictionPtr_0));
			xmm_sad = simde_mm_add_epi32(xmm_sad, simde_mm_sad_epu8(simde_mm_loadu_si128((simde__m128i*)(srcPtr + stride)), simde_mm_or_si128(simde_mm_and_si128(xmm_left, xmm_mask2), xmm_predictionDcValue)));
			xmm_left = simde_mm_srli_si128(xmm_left, 1);
			xmm_sad = simde_mm_add_epi32(xmm_sad, simde_mm_sad_epu8(simde_mm_loadu_si128((simde__m128i*)(srcPtr + (stride << 1))), simde_mm_or_si128(simde_mm_and_si128(xmm_left, xmm_mask2), xmm_predictionDcValue)));
			xmm_left = simde_mm_srli_si128(xmm_left, 1);
			xmm_sad = simde_mm_add_epi32(xmm_sad, simde_mm_sad_epu8(simde_mm_loadu_si128((simde__m128i*)(srcPtr + 3 * stride)), simde_mm_or_si128(simde_mm_and_si128(xmm_left, xmm_mask2), xmm_predictionDcValue)));
			xmm_left = simde_mm_srli_si128(xmm_left, 1);

			srcPtr += (stride << 2);

			for (idx = 4; idx < blockSize; idx += 4){
				xmm_sad = simde_mm_add_epi32(xmm_sad, simde_mm_sad_epu8(simde_mm_loadu_si128((simde__m128i*)(srcPtr)), simde_mm_or_si128(simde_mm_and_si128(xmm_left, xmm_mask2), xmm_predictionDcValue)));
				xmm_left = simde_mm_srli_si128(xmm_left, 1);
				xmm_sad = simde_mm_add_epi32(xmm_sad, simde_mm_sad_epu8(simde_mm_loadu_si128((simde__m128i*)(srcPtr + stride)), simde_mm_or_si128(simde_mm_and_si128(xmm_left, xmm_mask2), xmm_predictionDcValue)));
				xmm_left = simde_mm_srli_si128(xmm_left, 1);
				xmm_sad = simde_mm_add_epi32(xmm_sad, simde_mm_sad_epu8(simde_mm_loadu_si128((simde__m128i*)(srcPtr + (stride << 1))), simde_mm_or_si128(simde_mm_and_si128(xmm_left, xmm_mask2), xmm_predictionDcValue)));
				xmm_left = simde_mm_srli_si128(xmm_left, 1);
				xmm_sad = simde_mm_add_epi32(xmm_sad, simde_mm_sad_epu8(simde_mm_loadu_si128((simde__m128i*)(srcPtr + 3 * stride)), simde_mm_or_si128(simde_mm_and_si128(xmm_left, xmm_mask2), xmm_predictionDcValue)));
				xmm_left = simde_mm_srli_si128(xmm_left, 1);

				srcPtr += (stride << 2);

			}

			return simde_mm_cvtsi128_si32(xmm_sad);

		}
	}



	/*************************************************************************************************************************************************************************************************************/
	else {
		simde__m256i xmm_sum, xmm_sadleft, xmm_sadtop, xmm_toptmp, xmm_lefttmp, xmm_set, xmm_sum128_2, xmm_sum256, xmm_predictionDcValue;
		simde__m128i xmm_sumhi, xmm_sumlo, xmm_sum1, xmm_sum128, xmm_sumhitmp, xmm_sumlotmp, xmm_movelotmp, xmm_movehitmp;

		xmm_sumhi = xmm_sumlo = xmm_sum128 = xmm_sumhitmp = xmm_sumlotmp = simde_mm_setzero_si128();
		xmm_sum = xmm_sadleft = xmm_sadtop = xmm_toptmp = xmm_lefttmp = simde_mm256_setzero_si256();

		if (srcOriginY != 0)
		{
			xmm_toptmp = simde_mm256_sad_epu8(simde_mm256_set_m128i(simde_mm_loadu_si128((simde__m128i *)(srcPtr - stride + 16)), simde_mm_loadu_si128((simde__m128i *)(srcPtr - stride))), xmm1);

		}
		else
		{
			xmm_toptmp = simde_mm256_sad_epu8(simde_mm256_set_m128i(simde_mm_loadu_si128((simde__m128i *)(yBorderReverse + topOffset + 16)), simde_mm_loadu_si128((simde__m128i *)(yBorderReverse + topOffset))), xmm1);

		}
		if (srcOriginX != 0) {
			xmm_lefttmp = simde_mm256_sad_epu8(simde_mm256_set_epi8(*(readPtr + 31 * stride), *(readPtr + 30 * stride), *(readPtr + 29 * stride), *(readPtr + 28 * stride), *(readPtr + 27 * stride), *(readPtr + 26 * stride), *(readPtr + 25 * stride), *(readPtr + 24 * stride), *(readPtr + 23 * stride), *(readPtr + 22 * stride), *(readPtr + 21 * stride), *(readPtr + 20 * stride),
				*(readPtr + 19 * stride), *(readPtr + 18 * stride), *(readPtr + 17 * stride), *(readPtr + 16 * stride), *(readPtr + 15 * stride), *(readPtr + 14 * stride), *(readPtr + 13 * stride), *(readPtr + 12 * stride), *(readPtr + 11 * stride), *(readPtr + 10 * stride), *(readPtr + 9 * stride), *(readPtr + 8 * stride), *(readPtr + 7 * stride), *(readPtr + 6 * stride), *(readPtr + 5 * stride), *(readPtr + 4 * stride), *(readPtr + 3 * stride), *(readPtr + 2 * stride), *(readPtr + stride), *readPtr)
				, xmm1);

		}
		else
		{
			xmm_lefttmp = simde_mm256_sad_epu8(simde_mm256_set_m128i(simde_mm_loadu_si128((simde__m128i *)(yBorderReverse + leftOffset + 16)), simde_mm_loadu_si128((simde__m128i *)(yBorderReverse + leftOffset))), xmm1);

		}

		xmm_sum = simde_mm256_add_epi32(xmm_toptmp, xmm_lefttmp);
		xmm_sum = simde_mm256_hadd_epi32(xmm_sum, xmm_sum);

		xmm_sumlo = simde_mm256_extracti128_si256(xmm_sum, 0);
		xmm_sumhi = simde_mm256_extracti128_si256(xmm_sum, 1);

		xmm_movelotmp = simde_mm_move_epi64(xmm_sumlo);
		xmm_movehitmp = simde_mm_move_epi64(xmm_sumhi);

		xmm_sum1 = simde_mm_add_epi32(xmm_movelotmp, xmm_movehitmp);

		xmm_sum1 = simde_mm_hadd_epi32(xmm_sum1, xmm_sum1);

		xmm_sum256 = simde_mm256_castsi128_si256(xmm_sum1);


		xmm_set = simde_mm256_castsi128_si256(simde_mm_set1_epi32(32));

		xmm_sum128_2 = simde_mm256_add_epi32(xmm_sum256, xmm_set); // add offset
		xmm_predictionDcValue = simde_mm256_srli_epi32(xmm_sum128_2, 6); //simde_mm256_srli_epi32


		simde__m128i dc128 = simde_mm256_castsi256_si128(xmm_predictionDcValue);

		EB_U8 dc = simde_mm_cvtsi128_si32(dc128);
		xmm_predictionDcValue = simde_mm256_set1_epi8(dc);//simde_mm_broadcastb_epi8


		// SAD
		ymm0 = simde_mm256_setzero_si256();
		for (idx = 0; idx < blockSize; idx += 2) {
			ymm0 = simde_mm256_add_epi32(ymm0, simde_mm256_sad_epu8(simde_mm256_loadu_si256((simde__m256i*)srcPtr), xmm_predictionDcValue));
			xmm1 = simde_mm256_add_epi32(xmm1, simde_mm256_sad_epu8(simde_mm256_loadu_si256((simde__m256i*)(srcPtr + stride)), xmm_predictionDcValue));
			srcPtr += stride << 1;
		}
		ymm0 = simde_mm256_add_epi32(ymm0, xmm1);
		xmm0 = simde_mm_add_epi32(simde_mm256_extracti128_si256(ymm0, 0), simde_mm256_extracti128_si256(ymm0, 1));
		xmm0 = simde_mm_add_epi32(xmm0, simde_mm_srli_si128(xmm0, 8));
		return (EB_U32)simde_mm_cvtsi128_si32(xmm0);


	}
}

/***********************************************************************************************************************************************************************************************
                                                                        IntraModeAngular_18_AVX2_INTRIN
***********************************************************************************************************************************************************************************************/
void IntraModeAngular_18_AVX2_INTRIN(
    const EB_U32      size,                       //input parameter, denotes the size of the current PU
    EB_U8            *refSamples,                 //input parameter, pointer to the reference samples
    EB_U8            *predictionPtr,              //output parameter, pointer to the prediction
    const EB_U32      predictionBufferStride,     //input parameter, denotes the stride for the prediction ptr
    const EB_BOOL     skip)                    //skip one row 
{
    EB_U32 pStride = predictionBufferStride;
    EB_U32 topLeftOffset = (size << 1);

    if (!skip) {

        if (size == 32) {

            EB_U32 count; 
            for (count = 0; count < 8; ++count){

                simde_mm256_storeu_si256((simde__m256i *)predictionPtr,                      simde_mm256_loadu_si256((simde__m256i *)(refSamples + topLeftOffset)));
                simde_mm256_storeu_si256((simde__m256i *)(predictionPtr + pStride),          simde_mm256_loadu_si256((simde__m256i *)(refSamples + topLeftOffset - 1)));
                simde_mm256_storeu_si256((simde__m256i *)(predictionPtr + 2 * pStride),      simde_mm256_loadu_si256((simde__m256i *)(refSamples + topLeftOffset - 2)));
                simde_mm256_storeu_si256((simde__m256i *)(predictionPtr + 3 * pStride),      simde_mm256_loadu_si256((simde__m256i *)(refSamples + topLeftOffset - 3)));

                refSamples -= 4;
                predictionPtr += (pStride << 2);
            }
        }
        else if (size == 16) {
            simde_mm_storeu_si128((simde__m128i *)predictionPtr,                 simde_mm_loadu_si128((simde__m128i *)(refSamples + topLeftOffset )));
            simde_mm_storeu_si128((simde__m128i *)(predictionPtr + pStride),     simde_mm_loadu_si128((simde__m128i *)(refSamples + topLeftOffset - 1)));
            simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 2 * pStride), simde_mm_loadu_si128((simde__m128i *)(refSamples + topLeftOffset - 2)));
            simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 3 * pStride), simde_mm_loadu_si128((simde__m128i *)(refSamples + topLeftOffset - 3)));
            predictionPtr += (pStride << 2);
            simde_mm_storeu_si128((simde__m128i *)predictionPtr,                 simde_mm_loadu_si128((simde__m128i *)(refSamples + topLeftOffset - 4)));
            simde_mm_storeu_si128((simde__m128i *)(predictionPtr + pStride),     simde_mm_loadu_si128((simde__m128i *)(refSamples + topLeftOffset - 5)));
            simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 2 * pStride), simde_mm_loadu_si128((simde__m128i *)(refSamples + topLeftOffset - 6)));
            simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 3 * pStride), simde_mm_loadu_si128((simde__m128i *)(refSamples + topLeftOffset - 7)));
            predictionPtr += (pStride << 2);
            simde_mm_storeu_si128((simde__m128i *)predictionPtr,                 simde_mm_loadu_si128((simde__m128i *)(refSamples + topLeftOffset - 8)));
            simde_mm_storeu_si128((simde__m128i *)(predictionPtr + pStride),     simde_mm_loadu_si128((simde__m128i *)(refSamples + topLeftOffset - 9)));
            simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 2 * pStride), simde_mm_loadu_si128((simde__m128i *)(refSamples + topLeftOffset - 10)));
            simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 3 * pStride), simde_mm_loadu_si128((simde__m128i *)(refSamples + topLeftOffset - 11)));
            predictionPtr += (pStride << 2);
            simde_mm_storeu_si128((simde__m128i *)predictionPtr,                 simde_mm_loadu_si128((simde__m128i *)(refSamples + topLeftOffset - 12)));
            simde_mm_storeu_si128((simde__m128i *)(predictionPtr + pStride),     simde_mm_loadu_si128((simde__m128i *)(refSamples + topLeftOffset - 13)));
            simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 2 * pStride), simde_mm_loadu_si128((simde__m128i *)(refSamples + topLeftOffset - 14)));
            simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 3 * pStride), simde_mm_loadu_si128((simde__m128i *)(refSamples + topLeftOffset - 15)));
        }
        else if (size == 8) {
            simde_mm_storel_epi64((simde__m128i *)predictionPtr,                 simde_mm_loadl_epi64((simde__m128i *)(refSamples + topLeftOffset)));
            simde_mm_storel_epi64((simde__m128i *)(predictionPtr + pStride),     simde_mm_loadl_epi64((simde__m128i *)(refSamples + topLeftOffset - 1)));
            simde_mm_storel_epi64((simde__m128i *)(predictionPtr + 2 * pStride), simde_mm_loadl_epi64((simde__m128i *)(refSamples + topLeftOffset - 2)));
            simde_mm_storel_epi64((simde__m128i *)(predictionPtr + 3 * pStride), simde_mm_loadl_epi64((simde__m128i *)(refSamples + topLeftOffset - 3)));
            predictionPtr += (pStride << 2);
            simde_mm_storel_epi64((simde__m128i *)predictionPtr,                 simde_mm_loadl_epi64((simde__m128i *)(refSamples + topLeftOffset - 4)));
            simde_mm_storel_epi64((simde__m128i *)(predictionPtr + pStride),     simde_mm_loadl_epi64((simde__m128i *)(refSamples + topLeftOffset - 5)));
            simde_mm_storel_epi64((simde__m128i *)(predictionPtr + 2 * pStride), simde_mm_loadl_epi64((simde__m128i *)(refSamples + topLeftOffset - 6)));
            simde_mm_storel_epi64((simde__m128i *)(predictionPtr + 3 * pStride), simde_mm_loadl_epi64((simde__m128i *)(refSamples + topLeftOffset - 7)));
        }
        else {
            *(EB_U32 *)predictionPtr = *(EB_U32 *)(refSamples + topLeftOffset );
            *(EB_U32 *)(predictionPtr + pStride) = *(EB_U32 *)(refSamples + topLeftOffset - 1);
            *(EB_U32 *)(predictionPtr + 2 * pStride) = *(EB_U32 *)(refSamples + topLeftOffset -2);
            *(EB_U32 *)(predictionPtr + 3 * pStride) = *(EB_U32 *)(refSamples + topLeftOffset - 3);
        }
    }
    else {
        if (size != 4) {
            pStride <<= 1;

            if (size == 32) {
                EB_U32 count;

                for (count = 0; count < 4; ++count) {
                    
                    simde_mm256_storeu_si256((simde__m256i *)predictionPtr,                      simde_mm256_loadu_si256((simde__m256i *)(refSamples + topLeftOffset )));
                    simde_mm256_storeu_si256((simde__m256i *)(predictionPtr + pStride),          simde_mm256_loadu_si256((simde__m256i *)(refSamples + topLeftOffset - 2)));                    
                    simde_mm256_storeu_si256((simde__m256i *)(predictionPtr + 2 * pStride),      simde_mm256_loadu_si256((simde__m256i *)(refSamples + topLeftOffset - 4)));                    
                    simde_mm256_storeu_si256((simde__m256i *)(predictionPtr + 3 * pStride),      simde_mm256_loadu_si256((simde__m256i *)(refSamples + topLeftOffset - 6)));

                    refSamples -= 8;
                    predictionPtr += (pStride << 2);
                }
            }
            else if (size == 16) {
                simde_mm_storeu_si128((simde__m128i *)predictionPtr,                 simde_mm_loadu_si128((simde__m128i *)(refSamples + topLeftOffset)));
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + pStride),     simde_mm_loadu_si128((simde__m128i *)(refSamples + topLeftOffset - 2)));
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 2 * pStride), simde_mm_loadu_si128((simde__m128i *)(refSamples + topLeftOffset - 4)));
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 3 * pStride), simde_mm_loadu_si128((simde__m128i *)(refSamples + topLeftOffset - 6)));
                predictionPtr += (pStride << 2);
                simde_mm_storeu_si128((simde__m128i *)predictionPtr,                 simde_mm_loadu_si128((simde__m128i *)(refSamples + topLeftOffset - 8)));
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + pStride),     simde_mm_loadu_si128((simde__m128i *)(refSamples + topLeftOffset - 10)));
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 2 * pStride), simde_mm_loadu_si128((simde__m128i *)(refSamples + topLeftOffset - 12)));
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 3 * pStride), simde_mm_loadu_si128((simde__m128i *)(refSamples + topLeftOffset - 14)));
            }
            else {
                simde_mm_storel_epi64((simde__m128i *)predictionPtr,                 simde_mm_loadl_epi64((simde__m128i *)(refSamples + topLeftOffset )));
                simde_mm_storel_epi64((simde__m128i *)(predictionPtr + pStride),     simde_mm_loadl_epi64((simde__m128i *)(refSamples + topLeftOffset - 2)));
                simde_mm_storel_epi64((simde__m128i *)(predictionPtr + 2 * pStride), simde_mm_loadl_epi64((simde__m128i *)(refSamples + topLeftOffset - 4)));
                simde_mm_storel_epi64((simde__m128i *)(predictionPtr + 3 * pStride), simde_mm_loadl_epi64((simde__m128i *)(refSamples + topLeftOffset - 6)));                
            }
        }
        else {
            *(EB_U32*)predictionPtr = *(EB_U32*)(refSamples + topLeftOffset );
            *(EB_U32*)(predictionPtr + 2 * pStride) = *(EB_U32*)(refSamples + topLeftOffset - 2);
        }
    }
}

/*********************************************************************************************************************************************************************************************
                                                                IntraModeAngular_34_AVX2_INTRIN
***********************************************************************************************************************************************************************************************/

void IntraModeAngular_34_AVX2_INTRIN(
    const EB_U32      size,                       //input parameter, denotes the size of the current PU
    EB_U8            *refSamples,                 //input parameter, pointer to the reference samples
    EB_U8            *predictionPtr,              //output parameter, pointer to the prediction
    const EB_U32      predictionBufferStride,     //input parameter, denotes the stride for the prediction ptr
    const EB_BOOL     skip)                       //skip one row 
{
    EB_U32 pStride = predictionBufferStride;
    EB_U32 topOffset = ((size << 1) + 1);

    if (!skip) {

        if (size == 32) {
            
            EB_U32 count; 
            for (count = 0; count < 8; ++count){
                
                simde_mm256_storeu_si256((simde__m256i *)predictionPtr,                      simde_mm256_loadu_si256((simde__m256i *)(refSamples + topOffset + 1)));                
                simde_mm256_storeu_si256((simde__m256i *)(predictionPtr + pStride),          simde_mm256_loadu_si256((simde__m256i *)(refSamples + topOffset + 2)));                
                simde_mm256_storeu_si256((simde__m256i *)(predictionPtr + 2 * pStride),      simde_mm256_loadu_si256((simde__m256i *)(refSamples + topOffset + 3)));             
                simde_mm256_storeu_si256((simde__m256i *)(predictionPtr + 3 * pStride),      simde_mm256_loadu_si256((simde__m256i *)(refSamples + topOffset + 4)));

                refSamples += 4;
                predictionPtr += (pStride << 2);
            }
        }
        else if (size == 16) {
            simde_mm_storeu_si128((simde__m128i *)predictionPtr,                 simde_mm_loadu_si128((simde__m128i *)(refSamples + topOffset + 1)));
            simde_mm_storeu_si128((simde__m128i *)(predictionPtr + pStride),     simde_mm_loadu_si128((simde__m128i *)(refSamples + topOffset + 2)));
            simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 2 * pStride), simde_mm_loadu_si128((simde__m128i *)(refSamples + topOffset + 3)));
            simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 3 * pStride), simde_mm_loadu_si128((simde__m128i *)(refSamples + topOffset + 4)));
            predictionPtr += (pStride << 2);
            simde_mm_storeu_si128((simde__m128i *)predictionPtr,                 simde_mm_loadu_si128((simde__m128i *)(refSamples + topOffset + 5)));
            simde_mm_storeu_si128((simde__m128i *)(predictionPtr + pStride),     simde_mm_loadu_si128((simde__m128i *)(refSamples + topOffset + 6)));
            simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 2 * pStride), simde_mm_loadu_si128((simde__m128i *)(refSamples + topOffset + 7)));
            simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 3 * pStride), simde_mm_loadu_si128((simde__m128i *)(refSamples + topOffset + 8)));
            predictionPtr += (pStride << 2);
            simde_mm_storeu_si128((simde__m128i *)predictionPtr,                 simde_mm_loadu_si128((simde__m128i *)(refSamples + topOffset + 9)));
            simde_mm_storeu_si128((simde__m128i *)(predictionPtr + pStride),     simde_mm_loadu_si128((simde__m128i *)(refSamples + topOffset + 10)));
            simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 2 * pStride), simde_mm_loadu_si128((simde__m128i *)(refSamples + topOffset + 11)));
            simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 3 * pStride), simde_mm_loadu_si128((simde__m128i *)(refSamples + topOffset + 12)));
            predictionPtr += (pStride << 2);
            simde_mm_storeu_si128((simde__m128i *)predictionPtr,                 simde_mm_loadu_si128((simde__m128i *)(refSamples + topOffset + 13)));
            simde_mm_storeu_si128((simde__m128i *)(predictionPtr + pStride),     simde_mm_loadu_si128((simde__m128i *)(refSamples + topOffset + 14)));
            simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 2 * pStride), simde_mm_loadu_si128((simde__m128i *)(refSamples + topOffset + 15)));
            simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 3 * pStride), simde_mm_loadu_si128((simde__m128i *)(refSamples + topOffset + 16)));
        }
        else if (size == 8) {
            simde_mm_storel_epi64((simde__m128i *)predictionPtr,                 simde_mm_loadl_epi64((simde__m128i *)(refSamples + topOffset + 1)));
            simde_mm_storel_epi64((simde__m128i *)(predictionPtr + pStride),     simde_mm_loadl_epi64((simde__m128i *)(refSamples + topOffset + 2)));
            simde_mm_storel_epi64((simde__m128i *)(predictionPtr + 2 * pStride), simde_mm_loadl_epi64((simde__m128i *)(refSamples + topOffset + 3)));
            simde_mm_storel_epi64((simde__m128i *)(predictionPtr + 3 * pStride), simde_mm_loadl_epi64((simde__m128i *)(refSamples + topOffset + 4)));
            predictionPtr += (pStride << 2);
            simde_mm_storel_epi64((simde__m128i *)predictionPtr,                 simde_mm_loadl_epi64((simde__m128i *)(refSamples + topOffset + 5)));
            simde_mm_storel_epi64((simde__m128i *)(predictionPtr + pStride),     simde_mm_loadl_epi64((simde__m128i *)(refSamples + topOffset + 6)));
            simde_mm_storel_epi64((simde__m128i *)(predictionPtr + 2 * pStride), simde_mm_loadl_epi64((simde__m128i *)(refSamples + topOffset + 7)));
            simde_mm_storel_epi64((simde__m128i *)(predictionPtr + 3 * pStride), simde_mm_loadl_epi64((simde__m128i *)(refSamples + topOffset + 8)));
        }
        else {
            *(EB_U32 *)predictionPtr = *(EB_U32 *)(refSamples + topOffset + 1);
            *(EB_U32 *)(predictionPtr + pStride) = *(EB_U32 *)(refSamples + topOffset + 2);
            *(EB_U32 *)(predictionPtr + 2 * pStride) = *(EB_U32 *)(refSamples + topOffset + 3);
            *(EB_U32 *)(predictionPtr + 3 * pStride) = *(EB_U32 *)(refSamples + topOffset + 4);
        }
    }
    else {
        if (size != 4) {
            pStride <<= 1;

            if (size == 32) {
                EB_U32 count;

                for (count = 0; count < 4; ++count) {
                    simde_mm256_storeu_si256((simde__m256i *)predictionPtr,                      simde_mm256_loadu_si256((simde__m256i *)(refSamples + topOffset + 1)));
                    simde_mm256_storeu_si256((simde__m256i *)(predictionPtr + pStride),          simde_mm256_loadu_si256((simde__m256i *)(refSamples + topOffset + 3)));
                    simde_mm256_storeu_si256((simde__m256i *)(predictionPtr + 2 * pStride),      simde_mm256_loadu_si256((simde__m256i *)(refSamples + topOffset + 5)));
                    simde_mm256_storeu_si256((simde__m256i *)(predictionPtr + 3 * pStride),      simde_mm256_loadu_si256((simde__m256i *)(refSamples + topOffset + 7)));

                    refSamples += 8;
                    predictionPtr += (pStride << 2);
                }
            }
            else if (size == 16) {

                simde_mm_storeu_si128((simde__m128i *)predictionPtr,                 simde_mm_loadu_si128((simde__m128i *)(refSamples + topOffset + 1)));
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + pStride),     simde_mm_loadu_si128((simde__m128i *)(refSamples + topOffset + 3)));
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 2 * pStride), simde_mm_loadu_si128((simde__m128i *)(refSamples + topOffset + 5)));
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 3 * pStride), simde_mm_loadu_si128((simde__m128i *)(refSamples + topOffset + 7)));
                predictionPtr += (pStride << 2);
                simde_mm_storeu_si128((simde__m128i *)predictionPtr,                 simde_mm_loadu_si128((simde__m128i *)(refSamples + topOffset + 9)));
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + pStride),     simde_mm_loadu_si128((simde__m128i *)(refSamples + topOffset + 11)));
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 2 * pStride), simde_mm_loadu_si128((simde__m128i *)(refSamples + topOffset + 13)));
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 3 * pStride), simde_mm_loadu_si128((simde__m128i *)(refSamples + topOffset + 15)));
            }
            else {
                simde_mm_storel_epi64((simde__m128i *)predictionPtr,                 simde_mm_loadl_epi64((simde__m128i *)(refSamples + topOffset + 1)));
                simde_mm_storel_epi64((simde__m128i *)(predictionPtr + pStride),     simde_mm_loadl_epi64((simde__m128i *)(refSamples + topOffset + 3)));
                simde_mm_storel_epi64((simde__m128i *)(predictionPtr + 2 * pStride), simde_mm_loadl_epi64((simde__m128i *)(refSamples + topOffset + 5)));
                simde_mm_storel_epi64((simde__m128i *)(predictionPtr + 3 * pStride), simde_mm_loadl_epi64((simde__m128i *)(refSamples + topOffset + 7)));                
            }
        }
        else {
            *(EB_U32*)predictionPtr = *(EB_U32*)(refSamples + topOffset + 1);
            *(EB_U32*)(predictionPtr + 2 * pStride) = *(EB_U32*)(refSamples + topOffset + 3);
        }
    }
}

void IntraModeVerticalChroma_AVX2_INTRIN(
    const EB_U32      size,                   //input parameter, denotes the size of the current PU
    EB_U8            *refSamples,             //input parameter, pointer to the reference samples
    EB_U8            *predictionPtr,          //output parameter, pointer to the prediction
    const EB_U32      predictionBufferStride, //input parameter, denotes the stride for the prediction ptr
    const EB_BOOL     skip)                    //skip one row 
{
    EB_U32 pStride = predictionBufferStride;
    EB_U32 topOffset = (size << 1) + 1;

    // Jing: 
    // TODO: add size == 32 for 444
    if (!skip) {
        if (size == 32) {
            simde__m256i xmm0;
            EB_U64 size_to_write;
            EB_U32 count;

            // Each storeu calls stores 32 bytes. Hence each iteration stores 8 * 32 bytes.
            // Depending on skip, we need 4 or 2 iterations to store 32x32 bytes.
            size_to_write = 4 >> (skip ? 1 : 0);
            pStride = pStride << (skip ? 1 : 0);

            xmm0 = simde_mm256_loadu_si256((simde__m256i *)(refSamples + topOffset));

            for (count = 0; count < size_to_write; count ++) {
                simde_mm256_storeu_si256((simde__m256i *)(predictionPtr), xmm0);                    
                simde_mm256_storeu_si256((simde__m256i *)(predictionPtr + pStride), xmm0);          
                simde_mm256_storeu_si256((simde__m256i *)(predictionPtr + 2 * pStride), xmm0);      
                simde_mm256_storeu_si256((simde__m256i *)(predictionPtr + 3 * pStride), xmm0);         

                predictionPtr += (pStride << 2);                                          
                simde_mm256_storeu_si256((simde__m256i *)(predictionPtr), xmm0);                    
                simde_mm256_storeu_si256((simde__m256i *)(predictionPtr + pStride), xmm0);          
                simde_mm256_storeu_si256((simde__m256i *)(predictionPtr + 2 * pStride), xmm0);      
                simde_mm256_storeu_si256((simde__m256i *)(predictionPtr + 3 * pStride), xmm0);         

                predictionPtr += (pStride << 2);                                          
            }
        } else if (size == 16) {
            simde__m128i xmm0 = simde_mm_loadu_si128((simde__m128i *)(refSamples + topOffset)); 
            simde_mm_storeu_si128((simde__m128i *)predictionPtr, xmm0);                    
            simde_mm_storeu_si128((simde__m128i *)(predictionPtr + pStride), xmm0);        
            simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 2 * pStride), xmm0);    
            simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 3 * pStride), xmm0);       
            predictionPtr = predictionPtr + (pStride << 2);                         
            simde_mm_storeu_si128((simde__m128i *)predictionPtr, xmm0);                    
            simde_mm_storeu_si128((simde__m128i *)(predictionPtr + pStride), xmm0);        
            simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 2 * pStride), xmm0);    
            simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 3 * pStride), xmm0);       
            predictionPtr = predictionPtr + (pStride << 2);                         
            simde_mm_storeu_si128((simde__m128i *)predictionPtr, xmm0);                    
            simde_mm_storeu_si128((simde__m128i *)(predictionPtr + pStride), xmm0);        
            simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 2 * pStride), xmm0);    
            simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 3 * pStride), xmm0);       
            predictionPtr = predictionPtr + (pStride << 2);                         
            simde_mm_storeu_si128((simde__m128i *)predictionPtr, xmm0);                    
            simde_mm_storeu_si128((simde__m128i *)(predictionPtr + pStride), xmm0);        
            simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 2 * pStride), xmm0);    
            simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 3 * pStride), xmm0);       
        }
        else if (size == 8) {
            simde__m128i xmm0 = simde_mm_loadl_epi64((simde__m128i *)(refSamples + topOffset)); 
            simde_mm_storel_epi64((simde__m128i *)(predictionPtr), xmm0);                  
            simde_mm_storel_epi64((simde__m128i *)(predictionPtr + pStride), xmm0);        
            simde_mm_storel_epi64((simde__m128i *)(predictionPtr + 2 * pStride), xmm0);    
            simde_mm_storel_epi64((simde__m128i *)(predictionPtr + 3 * pStride), xmm0);       
            predictionPtr = predictionPtr + (pStride << 2);                         
            simde_mm_storel_epi64((simde__m128i *)predictionPtr, xmm0);                    
            simde_mm_storel_epi64((simde__m128i *)(predictionPtr + pStride), xmm0);        
            simde_mm_storel_epi64((simde__m128i *)(predictionPtr + 2 * pStride), xmm0);    
            simde_mm_storel_epi64((simde__m128i *)(predictionPtr + 3 * pStride), xmm0);       
        }
        else {
            EB_U32 top = *(EB_U32*)(refSamples + topOffset);         
            *(EB_U32*)(predictionPtr) = top;
            *(EB_U32*)(predictionPtr + pStride) = top;
            *(EB_U32*)(predictionPtr + 2 * pStride) = top;
            *(EB_U32*)(predictionPtr + 3 * pStride) = top;
        }
    }
    else {
        pStride <<= 1;
        if (size == 32) {
            EB_U32 columnIndex, rowIndex;
            EB_U32 writeIndex;
            EB_U32 topOffset = (size << 1) + 1;
            EB_U32 rowStride = skip ? 2 : 1;

            for (columnIndex = 0; columnIndex < size; ++columnIndex) {
                writeIndex = columnIndex;
                for (rowIndex = 0; rowIndex < size; rowIndex += rowStride) {
                    predictionPtr[writeIndex] = refSamples[topOffset + columnIndex];
                    writeIndex += rowStride * predictionBufferStride;
                }
            }
        } else if (size == 16) {
            
            simde__m128i xmm0 = simde_mm_loadu_si128((simde__m128i *)(refSamples + topOffset)); 
            simde_mm_storeu_si128((simde__m128i *)(predictionPtr), xmm0);                  
            simde_mm_storeu_si128((simde__m128i *)(predictionPtr + pStride), xmm0);        
            simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 2 * pStride), xmm0);    
            simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 3 * pStride), xmm0);       
            predictionPtr = predictionPtr + (pStride << 2);                         
            simde_mm_storeu_si128((simde__m128i *)(predictionPtr), xmm0);                  
            simde_mm_storeu_si128((simde__m128i *)(predictionPtr + pStride), xmm0);        
            simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 2 * pStride), xmm0);    
            simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 3 * pStride), xmm0);       
        }
        else if (size == 8) {
            
            simde__m128i xmm0 = simde_mm_loadl_epi64((simde__m128i *)(refSamples + topOffset)); 
            simde_mm_storel_epi64((simde__m128i *)predictionPtr, xmm0);                    
            simde_mm_storel_epi64((simde__m128i *)(predictionPtr + pStride), xmm0);        
            simde_mm_storel_epi64((simde__m128i *)(predictionPtr + 2 * pStride), xmm0);    
            simde_mm_storel_epi64((simde__m128i *)(predictionPtr + 3 * pStride), xmm0);       
        }
        else {
            EB_U32 top = *(EB_U32*)(refSamples + topOffset); 
            *(EB_U32*)(predictionPtr) = top;
            *(EB_U32*)(predictionPtr + pStride) = top;
        }
    }
}

void IntraModeDCChroma_AVX2_INTRIN(
    const EB_U32      size,                       //input parameter, denotes the size of the current PU
    EB_U8            *refSamples,                 //input parameter, pointer to the reference samples
    EB_U8            *predictionPtr,              //output parameter, pointer to the prediction
    const EB_U32      predictionBufferStride,     //input parameter, denotes the stride for the prediction ptr
    const EB_BOOL     skip)                       //skip one row 
{
    simde__m128i xmm0 = simde_mm_setzero_si128();
    EB_U32 pStride = predictionBufferStride;
    EB_U32 topOffset = (size << 1) + 1;
    EB_U32 leftOffset = 0;
    
    //Jing:
    //TODO: add size == 32 for 444
    if (!skip) {
        if (size == 32) {
            simde__m256i xmm_sum,xmm_sadleft,xmm_sadtop,xmm_toptmp,xmm_lefttmp ,xmm_set,xmm_sum128_2,xmm_sum256,xmm_predictionDcValue;
            simde__m256i xmm1 = simde_mm256_setzero_si256();
            simde__m128i xmm_sumhi,xmm_sumlo,xmm_sum1,xmm_sum128,xmm_sumhitmp,xmm_sumlotmp,xmm_movelotmp,xmm_movehitmp;

            xmm_sumhi = xmm_sumlo = xmm_sum128 = xmm_sumhitmp = xmm_sumlotmp = simde_mm_setzero_si128();
            xmm_sum = xmm_sadleft = xmm_sadtop =  xmm_toptmp = xmm_lefttmp  = simde_mm256_setzero_si256();

            xmm_toptmp  =simde_mm256_sad_epu8( simde_mm256_set_m128i ( simde_mm_loadu_si128( (simde__m128i *)(refSamples + topOffset +16) ),simde_mm_loadu_si128((simde__m128i *)(refSamples + topOffset))),xmm1);
            xmm_lefttmp =simde_mm256_sad_epu8( simde_mm256_set_m128i( simde_mm_loadu_si128((simde__m128i *)(refSamples + leftOffset+16)),simde_mm_loadu_si128((simde__m128i *)(refSamples + leftOffset))),xmm1);

            xmm_sum = simde_mm256_add_epi32(xmm_toptmp, xmm_lefttmp  ) ;
            xmm_sum = simde_mm256_hadd_epi32 (xmm_sum,xmm_sum);
            xmm_sumlo =  simde_mm256_extracti128_si256(xmm_sum,0);
            xmm_sumhi =  simde_mm256_extracti128_si256(xmm_sum,1);

            xmm_movelotmp = simde_mm_move_epi64 (xmm_sumlo);
            xmm_movehitmp = simde_mm_move_epi64 (xmm_sumhi);

            xmm_sum1 =   simde_mm_add_epi32(xmm_movelotmp,xmm_movehitmp);

            xmm_sum1 = simde_mm_hadd_epi32(xmm_sum1,xmm_sum1);

            xmm_sum256 = simde_mm256_castsi128_si256(xmm_sum1);

            xmm_set = simde_mm256_castsi128_si256(simde_mm_set1_epi32(32));

            xmm_sum128_2 = simde_mm256_add_epi32(xmm_sum256, xmm_set); // add offset
            xmm_predictionDcValue = simde_mm256_srli_epi32(xmm_sum128_2,6); //simde_mm256_srli_epi32


            simde__m128i dc128      = simde_mm256_castsi256_si128(xmm_predictionDcValue); 

            EB_U8 dc         = simde_mm_cvtsi128_si32 (dc128);
            xmm_predictionDcValue = simde_mm256_set1_epi8(dc);//simde_mm_broadcastb_epi8


            EB_U32 count;

            for (count = 0; count < 2; ++count) {

                simde_mm256_storeu_si256((simde__m256i *) predictionPtr, xmm_predictionDcValue);         
                simde_mm256_storeu_si256((simde__m256i *)(predictionPtr + 1 * pStride), xmm_predictionDcValue);
                simde_mm256_storeu_si256((simde__m256i *)(predictionPtr + 2 * pStride), xmm_predictionDcValue);
                simde_mm256_storeu_si256((simde__m256i *)(predictionPtr + 3 * pStride), xmm_predictionDcValue);

                predictionPtr += (pStride << 2);

                simde_mm256_storeu_si256((simde__m256i *) predictionPtr, xmm_predictionDcValue);  
                simde_mm256_storeu_si256((simde__m256i *)(predictionPtr + 1 * pStride), xmm_predictionDcValue);
                simde_mm256_storeu_si256((simde__m256i *)(predictionPtr + 2 * pStride), xmm_predictionDcValue);
                simde_mm256_storeu_si256((simde__m256i *)(predictionPtr + 3 * pStride), xmm_predictionDcValue);

                predictionPtr += (pStride << 2);

                simde_mm256_storeu_si256((simde__m256i *) predictionPtr, xmm_predictionDcValue);         
                simde_mm256_storeu_si256((simde__m256i *)(predictionPtr + 1 * pStride), xmm_predictionDcValue);
                simde_mm256_storeu_si256((simde__m256i *)(predictionPtr + 2 * pStride), xmm_predictionDcValue);
                simde_mm256_storeu_si256((simde__m256i *)(predictionPtr + 3 * pStride), xmm_predictionDcValue);

                predictionPtr += (pStride << 2);

                simde_mm256_storeu_si256((simde__m256i *) predictionPtr, xmm_predictionDcValue);  
                simde_mm256_storeu_si256((simde__m256i *)(predictionPtr + 1 * pStride), xmm_predictionDcValue);
                simde_mm256_storeu_si256((simde__m256i *)(predictionPtr + 2 * pStride), xmm_predictionDcValue);
                simde_mm256_storeu_si256((simde__m256i *)(predictionPtr + 3 * pStride), xmm_predictionDcValue);

                predictionPtr += (pStride << 2);
            }
        } else if (size == 16) {
            simde__m128i sum, predictionDcValue;
            
            sum = simde_mm_add_epi32(simde_mm_sad_epu8(simde_mm_loadu_si128((simde__m128i *)(refSamples + topOffset)), xmm0),
                                simde_mm_sad_epu8(simde_mm_loadu_si128((simde__m128i *)(refSamples + leftOffset)), xmm0));

            predictionDcValue = simde_mm_srli_epi32(simde_mm_add_epi32(simde_mm_add_epi32(simde_mm_srli_si128(sum, 8), sum), simde_mm_cvtsi32_si128(16)), 5);
            predictionDcValue = simde_mm_unpacklo_epi8(predictionDcValue, predictionDcValue);
            predictionDcValue = simde_mm_unpacklo_epi16(predictionDcValue, predictionDcValue);
            predictionDcValue = simde_mm_unpacklo_epi32(predictionDcValue, predictionDcValue);
            predictionDcValue = simde_mm_unpacklo_epi64(predictionDcValue, predictionDcValue);

            simde_mm_storeu_si128((simde__m128i *)(predictionPtr), predictionDcValue);
            simde_mm_storeu_si128((simde__m128i *)(predictionPtr + pStride), predictionDcValue);
            simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 2 * pStride), predictionDcValue);
            simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 3 * pStride), predictionDcValue);
            predictionPtr += (pStride << 2);
            simde_mm_storeu_si128((simde__m128i *)(predictionPtr), predictionDcValue);
            simde_mm_storeu_si128((simde__m128i *)(predictionPtr + pStride), predictionDcValue);
            simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 2 * pStride), predictionDcValue);
            simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 3 * pStride), predictionDcValue);
            predictionPtr += (pStride << 2);
            simde_mm_storeu_si128((simde__m128i *)(predictionPtr), predictionDcValue);
            simde_mm_storeu_si128((simde__m128i *)(predictionPtr + pStride), predictionDcValue);
            simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 2 * pStride), predictionDcValue);
            simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 3 * pStride), predictionDcValue);
            predictionPtr += (pStride << 2);
            simde_mm_storeu_si128((simde__m128i *)(predictionPtr), predictionDcValue);
            simde_mm_storeu_si128((simde__m128i *)(predictionPtr + pStride), predictionDcValue);
            simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 2 * pStride), predictionDcValue);
            simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 3 * pStride), predictionDcValue);

        }
        else if (size == 8) {
            simde__m128i sum, predictionDcValue;
 
            sum = simde_mm_add_epi32(simde_mm_sad_epu8(simde_mm_loadl_epi64((simde__m128i *)(refSamples + topOffset)), xmm0), 
                                simde_mm_sad_epu8(simde_mm_loadl_epi64((simde__m128i *)(refSamples + leftOffset)), xmm0));
            
            predictionDcValue = simde_mm_srli_epi32(simde_mm_add_epi32(sum, simde_mm_cvtsi32_si128(8)), 4);
            predictionDcValue = simde_mm_unpacklo_epi8(predictionDcValue, predictionDcValue);
            predictionDcValue = simde_mm_unpacklo_epi16(predictionDcValue, predictionDcValue);
            predictionDcValue = simde_mm_unpacklo_epi32(predictionDcValue, predictionDcValue);

            simde_mm_storel_epi64((simde__m128i *)(predictionPtr), predictionDcValue);
            simde_mm_storel_epi64((simde__m128i *)(predictionPtr + pStride), predictionDcValue);
            simde_mm_storel_epi64((simde__m128i *)(predictionPtr + 2 * pStride), predictionDcValue);
            simde_mm_storel_epi64((simde__m128i *)(predictionPtr + 3 * pStride), predictionDcValue);
            predictionPtr += (pStride << 2);
            simde_mm_storel_epi64((simde__m128i *)(predictionPtr), predictionDcValue);
            simde_mm_storel_epi64((simde__m128i *)(predictionPtr + pStride), predictionDcValue);
            simde_mm_storel_epi64((simde__m128i *)(predictionPtr + 2 * pStride), predictionDcValue);
            simde_mm_storel_epi64((simde__m128i *)(predictionPtr + 3 * pStride), predictionDcValue);
        }
        else {
            simde__m128i sum, predictionDcValue;
            
            sum = simde_mm_add_epi32(simde_mm_sad_epu8(simde_mm_cvtsi32_si128(*(EB_U32*)(refSamples + topOffset)), xmm0), 
                                simde_mm_sad_epu8(simde_mm_cvtsi32_si128(*(EB_U32*)(refSamples + leftOffset)), xmm0));
           
            predictionDcValue = simde_mm_srli_epi32(simde_mm_add_epi32(sum, simde_mm_cvtsi32_si128(4)), 3);
            predictionDcValue = simde_mm_unpacklo_epi8(predictionDcValue, predictionDcValue);
            predictionDcValue = simde_mm_unpacklo_epi16(predictionDcValue, predictionDcValue);

            *(EB_U32*)predictionPtr =                 simde_mm_cvtsi128_si32(predictionDcValue);
            *(EB_U32*)(predictionPtr + pStride) =     simde_mm_cvtsi128_si32(predictionDcValue);
            *(EB_U32*)(predictionPtr + 2 * pStride) = simde_mm_cvtsi128_si32(predictionDcValue);
            *(EB_U32*)(predictionPtr + 3 * pStride) = simde_mm_cvtsi128_si32(predictionDcValue);
        }
    }
    else {

        pStride <<= 1;
        if (size == 32) {
            EB_U32 sum = 0;
            EB_U32 index;
            EB_U32 columnIndex, rowIndex;
            EB_U32 writeIndex;
            EB_U32 leftOffset = 0;
            EB_U32 topOffset = (size << 1) + 1;
            EB_U32 predictionDcValue = 128; // needs to be changed to a macro based on bit depth
            EB_U32 rowStride = skip ? 2 : 1;

            // top reference samples
            for (index = 0; index< size; index++) {
                sum += refSamples[topOffset + index];
            }

            // left reference samples
            for (index = 0; index< size; index++) {
                sum += refSamples[leftOffset + index];
            }

            predictionDcValue = (EB_U8)((sum + size) >> Log2f(size << 1));

            // Generate the prediction
            for (rowIndex = 0; rowIndex < size; rowIndex += rowStride) {
                writeIndex = rowIndex * predictionBufferStride;
                for (columnIndex = 0; columnIndex < size; ++columnIndex) {
                    predictionPtr[writeIndex] = (EB_U8)predictionDcValue;
                    ++writeIndex;
                }
            }

        } else if (size == 16) {

            simde__m128i sum, predictionDcValue;
            
            sum = simde_mm_add_epi32(simde_mm_sad_epu8(simde_mm_loadu_si128((simde__m128i *)(refSamples + topOffset)), xmm0),  
                                simde_mm_sad_epu8(simde_mm_loadu_si128((simde__m128i *)(refSamples + leftOffset)), xmm0));

            predictionDcValue = simde_mm_srli_epi32(simde_mm_add_epi32(simde_mm_add_epi32(simde_mm_srli_si128(sum, 8), sum), simde_mm_cvtsi32_si128(16)), 5);
            predictionDcValue = simde_mm_unpacklo_epi8(predictionDcValue, predictionDcValue);
            predictionDcValue = simde_mm_unpacklo_epi16(predictionDcValue, predictionDcValue);
            predictionDcValue = simde_mm_unpacklo_epi32(predictionDcValue, predictionDcValue);
            predictionDcValue = simde_mm_unpacklo_epi64(predictionDcValue, predictionDcValue);

            simde_mm_storeu_si128((simde__m128i *)(predictionPtr), predictionDcValue);
            simde_mm_storeu_si128((simde__m128i *)(predictionPtr + pStride), predictionDcValue);
            simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 2 * pStride), predictionDcValue);
            simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 3 * pStride), predictionDcValue);
            predictionPtr += (pStride << 2);
            simde_mm_storeu_si128((simde__m128i *)(predictionPtr), predictionDcValue);
            simde_mm_storeu_si128((simde__m128i *)(predictionPtr + pStride), predictionDcValue);
            simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 2 * pStride), predictionDcValue);
            simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 3 * pStride), predictionDcValue);
        }
        else if (size == 8) {
            simde__m128i sum, predictionDcValue;
            
            sum = simde_mm_add_epi32(simde_mm_sad_epu8(simde_mm_loadl_epi64((simde__m128i *)(refSamples + topOffset)), xmm0), 
                                simde_mm_sad_epu8(simde_mm_loadl_epi64((simde__m128i *)(refSamples + leftOffset)), xmm0));
            
            predictionDcValue = simde_mm_srli_epi32(simde_mm_add_epi32(sum, simde_mm_cvtsi32_si128(8)), 4);
            predictionDcValue = simde_mm_unpacklo_epi8(predictionDcValue, predictionDcValue);
            predictionDcValue = simde_mm_unpacklo_epi16(predictionDcValue, predictionDcValue);
            predictionDcValue = simde_mm_unpacklo_epi32(predictionDcValue, predictionDcValue);

            simde_mm_storel_epi64((simde__m128i *)(predictionPtr), predictionDcValue);
            simde_mm_storel_epi64((simde__m128i *)(predictionPtr + pStride), predictionDcValue);
            simde_mm_storel_epi64((simde__m128i *)(predictionPtr + 2 * pStride), predictionDcValue);
            simde_mm_storel_epi64((simde__m128i *)(predictionPtr + 3 * pStride), predictionDcValue);
        }
        else {
            simde__m128i sum, predictionDcValue;
            
            sum = simde_mm_add_epi32(simde_mm_sad_epu8(simde_mm_cvtsi32_si128(*(EB_U32*)(refSamples + topOffset)), xmm0), 
                                simde_mm_sad_epu8(simde_mm_cvtsi32_si128(*(EB_U32*)(refSamples + leftOffset)), xmm0));

            predictionDcValue = simde_mm_srli_epi32(simde_mm_add_epi32(sum, simde_mm_cvtsi32_si128(4)), 3);
            predictionDcValue = simde_mm_unpacklo_epi8(predictionDcValue, predictionDcValue);
            predictionDcValue = simde_mm_unpacklo_epi16(predictionDcValue, predictionDcValue);

            *(EB_U32*)predictionPtr =             simde_mm_cvtsi128_si32(predictionDcValue);
            *(EB_U32*)(predictionPtr + pStride) = simde_mm_cvtsi128_si32(predictionDcValue);
        }
    }
}

