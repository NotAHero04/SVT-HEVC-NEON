/*
* Copyright(c) 2018 Intel Corporation
* SPDX - License - Identifier: BSD - 2 - Clause - Patent
*/

#include "EbDefinitions.h"
#include "../../../simde/simde/x86/sse2.h"
#include "EbIntrinMacros16bit_SSE2.h"
#include "EbIntraPrediction_SSE2.h"


#define MACRO_VERTICAL_LUMA_16BIT_4(ARG1, ARG2, ARG3)\
    simde_mm_storel_epi64((simde__m128i *)predictionPtr, simde_mm_or_si128(simde_mm_and_si128(ARG1, ARG2), ARG3));\
    ARG1 = simde_mm_srli_si128(ARG1, 2);\
    simde_mm_storel_epi64((simde__m128i *)(predictionPtr + pStride), simde_mm_or_si128(simde_mm_and_si128(ARG1, ARG2), ARG3));\
    ARG1 = simde_mm_srli_si128(ARG1, 2);

#define MACRO_VERTICAL_LUMA_16(ARG1, ARG2, ARG3, ARG8, ARG9)\
    simde_mm_storeu_si128((simde__m128i *)predictionPtr, simde_mm_or_si128(simde_mm_and_si128(ARG1, ARG2), ARG3));\
    simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 8), ARG8);\
    ARG1 = simde_mm_srli_si128(ARG1, ARG9);\
    simde_mm_storeu_si128((simde__m128i *)(predictionPtr + pStride), simde_mm_or_si128(simde_mm_and_si128(ARG1, ARG2), ARG3));\
    simde_mm_storeu_si128((simde__m128i *)(predictionPtr + pStride + 8), ARG8);\
    ARG1 = simde_mm_srli_si128(ARG1, ARG9);\
    simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 2 * pStride), simde_mm_or_si128(simde_mm_and_si128(ARG1, ARG2), ARG3));\
    simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 2 * pStride + 8), ARG8);\
    ARG1 = simde_mm_srli_si128(ARG1, ARG9);\
    simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 3 * pStride), simde_mm_or_si128(simde_mm_and_si128(ARG1, ARG2), ARG3));\
    simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 3 * pStride + 8), ARG8);\
    ARG1 = simde_mm_srli_si128(ARG1, ARG9);

#define MACRO_HORIZONTAL_LUMA_32X16(A)\
    left0_14_even = simde_mm_packs_epi32(simde_mm_and_si128(simde_mm_loadu_si128((simde__m128i *)(refSamples+leftOffset+A)), skip_mask), \
                                    simde_mm_and_si128(simde_mm_loadu_si128((simde__m128i *)(refSamples+leftOffset+A+8)), skip_mask));\
    left_8_14_even = simde_mm_unpackhi_epi16(left0_14_even, left0_14_even);\
    left_0_6_even = simde_mm_unpacklo_epi16(left0_14_even, left0_14_even);\
    left46 = simde_mm_unpackhi_epi32(left_0_6_even, left_0_6_even);\
    left02 = simde_mm_unpacklo_epi32(left_0_6_even, left_0_6_even);\
    left12_14 = simde_mm_unpackhi_epi32(left_8_14_even, left_8_14_even);\
    left8_10 = simde_mm_unpacklo_epi32(left_8_14_even, left_8_14_even);\
    left0 = simde_mm_unpacklo_epi64(left02, left02);\
    left2 = simde_mm_unpackhi_epi64(left02, left02);\
    left4 = simde_mm_unpacklo_epi64(left46, left46);\
    left6 = simde_mm_unpackhi_epi64(left46, left46);\
    left8 = simde_mm_unpacklo_epi64(left8_10, left8_10);\
    left10 = simde_mm_unpackhi_epi64(left8_10, left8_10);\
    left12 = simde_mm_unpacklo_epi64(left12_14, left12_14);\
    left14 = simde_mm_unpackhi_epi64(left12_14, left12_14);\
    simde_mm_storeu_si128((simde__m128i *)(predictionPtr), left0);\
    simde_mm_storeu_si128((simde__m128i *)(predictionPtr+8), left0);\
    simde_mm_storeu_si128((simde__m128i *)(predictionPtr+16), left0);\
    simde_mm_storeu_si128((simde__m128i *)(predictionPtr+24), left0);\
    simde_mm_storeu_si128((simde__m128i *)(predictionPtr+pStride), left2);\
    simde_mm_storeu_si128((simde__m128i *)(predictionPtr+pStride+8), left2);\
    simde_mm_storeu_si128((simde__m128i *)(predictionPtr+pStride+16), left2);\
    simde_mm_storeu_si128((simde__m128i *)(predictionPtr+pStride+24), left2);\
    simde_mm_storeu_si128((simde__m128i *)(predictionPtr+2*pStride), left4);\
    simde_mm_storeu_si128((simde__m128i *)(predictionPtr+2*pStride+8), left4);\
    simde_mm_storeu_si128((simde__m128i *)(predictionPtr+2*pStride+16), left4);\
    simde_mm_storeu_si128((simde__m128i *)(predictionPtr+2*pStride+24), left4);\
    simde_mm_storeu_si128((simde__m128i *)(predictionPtr+3*pStride), left6);\
    simde_mm_storeu_si128((simde__m128i *)(predictionPtr+3*pStride+8), left6);\
    simde_mm_storeu_si128((simde__m128i *)(predictionPtr+3*pStride+16), left6);\
    simde_mm_storeu_si128((simde__m128i *)(predictionPtr+3*pStride+24), left6);\
    predictionPtr += (pStride << 2);\
    simde_mm_storeu_si128((simde__m128i *)(predictionPtr), left8);\
    simde_mm_storeu_si128((simde__m128i *)(predictionPtr+8), left8);\
    simde_mm_storeu_si128((simde__m128i *)(predictionPtr+16), left8);\
    simde_mm_storeu_si128((simde__m128i *)(predictionPtr+24), left8);\
    simde_mm_storeu_si128((simde__m128i *)(predictionPtr+pStride), left10);\
    simde_mm_storeu_si128((simde__m128i *)(predictionPtr+pStride+8), left10);\
    simde_mm_storeu_si128((simde__m128i *)(predictionPtr+pStride+16), left10);\
    simde_mm_storeu_si128((simde__m128i *)(predictionPtr+pStride+24), left10);\
    simde_mm_storeu_si128((simde__m128i *)(predictionPtr+2*pStride), left12);\
    simde_mm_storeu_si128((simde__m128i *)(predictionPtr+2*pStride+8), left12);\
    simde_mm_storeu_si128((simde__m128i *)(predictionPtr+2*pStride+16), left12);\
    simde_mm_storeu_si128((simde__m128i *)(predictionPtr+2*pStride+24), left12);\
    simde_mm_storeu_si128((simde__m128i *)(predictionPtr+3*pStride), left14);\
    simde_mm_storeu_si128((simde__m128i *)(predictionPtr+3*pStride+8), left14);\
    simde_mm_storeu_si128((simde__m128i *)(predictionPtr+3*pStride+16), left14);\
    simde_mm_storeu_si128((simde__m128i *)(predictionPtr+3*pStride+24), left14);

void IntraModeVerticalLuma16bit_SSE2_INTRIN(
    const EB_U32   size,
    EB_U16         *refSamples,
    EB_U16         *predictionPtr,
    const EB_U32   predictionBufferStride,
    const EB_BOOL  skip)
{
    EB_U32 leftOffset = 0;
    EB_U32 topLeftOffset = (size << 1);
    EB_U32 topOffset = ((size << 1) + 1);
    EB_U32 pStride = predictionBufferStride;
    
    if (size != 32) {
        simde__m128i xmm0 = simde_mm_setzero_si128();
        simde__m128i xmm_mask1 = simde_mm_slli_si128( simde_mm_set1_epi16((signed char)(0xFFFF)), 2);
        simde__m128i xmm_mask2 = simde_mm_srli_si128(xmm_mask1, 14);
        simde__m128i xmm_Max10bit = simde_mm_set1_epi16(0x03FF);
        simde__m128i xmm_topLeft_hi, xmm_topLeft_lo, xmm_topLeft, xmm_left, xmm_top;

        xmm_topLeft = simde_mm_cvtsi32_si128(*(EB_U32*)(refSamples + topLeftOffset));
        xmm_topLeft = simde_mm_unpacklo_epi16(xmm_topLeft, xmm_topLeft);
        xmm_topLeft = simde_mm_unpacklo_epi32(xmm_topLeft, xmm_topLeft);
        xmm_topLeft_hi = simde_mm_unpackhi_epi64(xmm_topLeft, xmm_topLeft);
        xmm_topLeft_lo = simde_mm_unpacklo_epi64(xmm_topLeft, xmm_topLeft);
        
        if (!skip) {

            if (size ==16) {
                simde__m128i xmm_left_lo, xmm_left_hi, xmm_top_lo, xmm_top_hi;
                xmm_left_lo = simde_mm_add_epi16(simde_mm_srai_epi16(simde_mm_sub_epi16(simde_mm_loadu_si128((simde__m128i *)(refSamples + leftOffset)), xmm_topLeft_lo), 1), xmm_topLeft_hi);
                xmm_left_hi = simde_mm_add_epi16(simde_mm_srai_epi16(simde_mm_sub_epi16(simde_mm_loadu_si128((simde__m128i *)(refSamples + leftOffset + 8)), xmm_topLeft_lo), 1), xmm_topLeft_hi);
                xmm_left_lo = simde_mm_max_epi16(simde_mm_min_epi16(xmm_left_lo, xmm_Max10bit), xmm0); 
                xmm_left_hi = simde_mm_max_epi16(simde_mm_min_epi16(xmm_left_hi, xmm_Max10bit), xmm0); 
                xmm_top_hi = simde_mm_loadu_si128((simde__m128i *)(refSamples + topOffset + 8));
                xmm_top_lo = simde_mm_and_si128(simde_mm_loadu_si128((simde__m128i *)(refSamples + topOffset)), xmm_mask1);
                MACRO_VERTICAL_LUMA_16(xmm_left_lo, xmm_mask2, xmm_top_lo, xmm_top_hi, 2)
                predictionPtr += (pStride << 2); 
                MACRO_VERTICAL_LUMA_16(xmm_left_lo, xmm_mask2, xmm_top_lo, xmm_top_hi, 2) 
                predictionPtr += (pStride << 2); 
                MACRO_VERTICAL_LUMA_16(xmm_left_hi, xmm_mask2, xmm_top_lo, xmm_top_hi, 2) 
                predictionPtr += (pStride << 2); 
                MACRO_VERTICAL_LUMA_16(xmm_left_hi, xmm_mask2, xmm_top_lo, xmm_top_hi, 2) 
            }
            else if (size == 8) {
                xmm_left = simde_mm_add_epi16(simde_mm_srai_epi16(simde_mm_sub_epi16(simde_mm_loadu_si128((simde__m128i *)(refSamples + leftOffset)), xmm_topLeft_lo), 1), xmm_topLeft_hi);
                xmm_left = simde_mm_max_epi16(simde_mm_min_epi16(xmm_left, xmm_Max10bit), xmm0);
                xmm_top = simde_mm_and_si128(simde_mm_loadu_si128((simde__m128i *)(refSamples + topOffset)), xmm_mask1);
                MACRO_VERTICAL_LUMA_8(xmm_left, xmm_mask2, xmm_top)
                predictionPtr += (pStride << 2);
                MACRO_VERTICAL_LUMA_8(xmm_left, xmm_mask2, xmm_top)
            }
            else {
                xmm_left = simde_mm_add_epi16(simde_mm_srai_epi16(simde_mm_sub_epi16(simde_mm_loadl_epi64((simde__m128i *)(refSamples + leftOffset)), xmm_topLeft_lo), 1), xmm_topLeft_hi);
                xmm_left = simde_mm_max_epi16(simde_mm_min_epi16(xmm_left, xmm_Max10bit), xmm0);
                xmm_top = simde_mm_and_si128 (simde_mm_loadl_epi64((simde__m128i *)(refSamples + topOffset)), xmm_mask1);
                MACRO_VERTICAL_LUMA_16BIT_4(xmm_left, xmm_mask2, xmm_top)
                predictionPtr += (pStride << 1);
                MACRO_VERTICAL_LUMA_16BIT_4(xmm_left, xmm_mask2, xmm_top)
            }
        }
        else {
            pStride <<= 1;

            simde__m128i xmm_mask_skip = simde_mm_set1_epi32(0x0000FFFF);

            if (size == 16) {
                simde__m128i xmm_top_lo, xmm_top_hi;
               
                xmm_left = simde_mm_packs_epi32(simde_mm_and_si128(simde_mm_loadu_si128((simde__m128i *)(refSamples + leftOffset)), xmm_mask_skip), simde_mm_and_si128(simde_mm_loadu_si128((simde__m128i *)(refSamples + leftOffset + 8)), xmm_mask_skip));
                xmm_left = simde_mm_max_epi16(simde_mm_min_epi16(simde_mm_add_epi16(simde_mm_srai_epi16( simde_mm_sub_epi16(xmm_left, xmm_topLeft_lo), 1), xmm_topLeft_hi), xmm_Max10bit), xmm0);
                xmm_top_hi = simde_mm_loadu_si128((simde__m128i *)(refSamples + topOffset + 8));
                xmm_top_lo = simde_mm_and_si128(simde_mm_loadu_si128((simde__m128i *)(refSamples + topOffset)), xmm_mask1);
                MACRO_VERTICAL_LUMA_16(xmm_left, xmm_mask2, xmm_top_lo, xmm_top_hi, 2)
                predictionPtr += (pStride << 2);
                MACRO_VERTICAL_LUMA_16(xmm_left, xmm_mask2, xmm_top_lo, xmm_top_hi, 2)
            }
            else {
                xmm_left = simde_mm_srai_epi16(simde_mm_sub_epi16(simde_mm_packs_epi32(simde_mm_and_si128(simde_mm_loadu_si128((simde__m128i *)(refSamples + leftOffset)), xmm_mask_skip), xmm0), xmm_topLeft_lo), 1);
                xmm_left = simde_mm_max_epi16(simde_mm_min_epi16(simde_mm_add_epi16(xmm_left, xmm_topLeft_hi), xmm_Max10bit), xmm0);
                xmm_top = simde_mm_and_si128(simde_mm_loadu_si128((simde__m128i *)(refSamples + topOffset)), xmm_mask1);
                MACRO_VERTICAL_LUMA_8(xmm_left, xmm_mask2, xmm_top)
            }
        }
    }
    else {
        simde__m128i top_0_7 = simde_mm_loadu_si128((simde__m128i *)(refSamples + topOffset));
        simde__m128i top_8_15 = simde_mm_loadu_si128((simde__m128i *)(refSamples + topOffset + 8));
        simde__m128i top_16_23 = simde_mm_loadu_si128((simde__m128i *)(refSamples + topOffset + 16));
        simde__m128i top_24_31 = simde_mm_loadu_si128((simde__m128i *)(refSamples + topOffset + 24));
        EB_U64 count, size_to_write;
        
        pStride <<= (skip ? 1 : 0);

        // Each 2 storeu calls stores 32 bytes. Hence each iteration stores 8 * 32 bytes.
        // Depending on skip, we need 4 or 2 iterations to store 32x32 bytes.
        size_to_write = 4 >> (skip ? 1 : 0);

        for (count = 0; count < size_to_write; ++count) {
            simde_mm_storeu_si128((simde__m128i *)(predictionPtr), top_0_7);
            simde_mm_storeu_si128((simde__m128i *)(predictionPtr+8), top_8_15);
            simde_mm_storeu_si128((simde__m128i *)(predictionPtr+16), top_16_23);
            simde_mm_storeu_si128((simde__m128i *)(predictionPtr+24), top_24_31);
            simde_mm_storeu_si128((simde__m128i *)(predictionPtr+pStride), top_0_7);
            simde_mm_storeu_si128((simde__m128i *)(predictionPtr+pStride+8), top_8_15);
            simde_mm_storeu_si128((simde__m128i *)(predictionPtr+pStride+16), top_16_23);
            simde_mm_storeu_si128((simde__m128i *)(predictionPtr+pStride+24), top_24_31);
            simde_mm_storeu_si128((simde__m128i *)(predictionPtr+2*pStride), top_0_7);
            simde_mm_storeu_si128((simde__m128i *)(predictionPtr+2*pStride+8), top_8_15);
            simde_mm_storeu_si128((simde__m128i *)(predictionPtr+2*pStride+16), top_16_23);
            simde_mm_storeu_si128((simde__m128i *)(predictionPtr+2*pStride+24), top_24_31);
            simde_mm_storeu_si128((simde__m128i *)(predictionPtr+3*pStride), top_0_7);
            simde_mm_storeu_si128((simde__m128i *)(predictionPtr+3*pStride+8), top_8_15);
            simde_mm_storeu_si128((simde__m128i *)(predictionPtr+3*pStride+16), top_16_23);
            simde_mm_storeu_si128((simde__m128i *)(predictionPtr+3*pStride+24), top_24_31);
            predictionPtr += (pStride << 2);
            simde_mm_storeu_si128((simde__m128i *)(predictionPtr), top_0_7);
            simde_mm_storeu_si128((simde__m128i *)(predictionPtr+8), top_8_15);
            simde_mm_storeu_si128((simde__m128i *)(predictionPtr+16), top_16_23);
            simde_mm_storeu_si128((simde__m128i *)(predictionPtr+24), top_24_31);
            simde_mm_storeu_si128((simde__m128i *)(predictionPtr+pStride), top_0_7);
            simde_mm_storeu_si128((simde__m128i *)(predictionPtr+pStride+8), top_8_15);
            simde_mm_storeu_si128((simde__m128i *)(predictionPtr+pStride+16), top_16_23);
            simde_mm_storeu_si128((simde__m128i *)(predictionPtr+pStride+24), top_24_31);
            simde_mm_storeu_si128((simde__m128i *)(predictionPtr+2*pStride), top_0_7);
            simde_mm_storeu_si128((simde__m128i *)(predictionPtr+2*pStride+8), top_8_15);
            simde_mm_storeu_si128((simde__m128i *)(predictionPtr+2*pStride+16), top_16_23);
            simde_mm_storeu_si128((simde__m128i *)(predictionPtr+2*pStride+24), top_24_31);
            simde_mm_storeu_si128((simde__m128i *)(predictionPtr+3*pStride), top_0_7);
            simde_mm_storeu_si128((simde__m128i *)(predictionPtr+3*pStride+8), top_8_15);
            simde_mm_storeu_si128((simde__m128i *)(predictionPtr+3*pStride+16), top_16_23);
            simde_mm_storeu_si128((simde__m128i *)(predictionPtr+3*pStride+24), top_24_31);
            predictionPtr += (pStride << 2);
        }
    }
    return;
}

void IntraModeVerticalChroma16bit_SSE2_INTRIN(
    const EB_U32   size,
    EB_U16         *refSamples,
    EB_U16         *predictionPtr,
    const EB_U32   predictionBufferStride,
    const EB_BOOL  skip)
{
    EB_U32 pStride = predictionBufferStride;
    EB_U32 topOffset = (size << 1) + 1;

    if (size == 32) {
        simde__m128i top_0_7 = simde_mm_loadu_si128((simde__m128i *)(refSamples + topOffset));
        simde__m128i top_8_15 = simde_mm_loadu_si128((simde__m128i *)(refSamples + topOffset + 8));
        simde__m128i top_16_23 = simde_mm_loadu_si128((simde__m128i *)(refSamples + topOffset + 16));
        simde__m128i top_24_31 = simde_mm_loadu_si128((simde__m128i *)(refSamples + topOffset + 24));
        EB_U64 count, size_to_write;

        pStride <<= (skip ? 1 : 0);

        // Each 2 storeu calls stores 32 bytes. Hence each iteration stores 8 * 32 bytes.
        // Depending on skip, we need 4 or 2 iterations to store 32x32 bytes.
        size_to_write = 4 >> (skip ? 1 : 0);

        for (count = 0; count < size_to_write; ++count) {
            simde_mm_storeu_si128((simde__m128i *)(predictionPtr), top_0_7);
            simde_mm_storeu_si128((simde__m128i *)(predictionPtr+8), top_8_15);
            simde_mm_storeu_si128((simde__m128i *)(predictionPtr+16), top_16_23);
            simde_mm_storeu_si128((simde__m128i *)(predictionPtr+24), top_24_31);
            simde_mm_storeu_si128((simde__m128i *)(predictionPtr+pStride), top_0_7);
            simde_mm_storeu_si128((simde__m128i *)(predictionPtr+pStride+8), top_8_15);
            simde_mm_storeu_si128((simde__m128i *)(predictionPtr+pStride+16), top_16_23);
            simde_mm_storeu_si128((simde__m128i *)(predictionPtr+pStride+24), top_24_31);
            simde_mm_storeu_si128((simde__m128i *)(predictionPtr+2*pStride), top_0_7);
            simde_mm_storeu_si128((simde__m128i *)(predictionPtr+2*pStride+8), top_8_15);
            simde_mm_storeu_si128((simde__m128i *)(predictionPtr+2*pStride+16), top_16_23);
            simde_mm_storeu_si128((simde__m128i *)(predictionPtr+2*pStride+24), top_24_31);
            simde_mm_storeu_si128((simde__m128i *)(predictionPtr+3*pStride), top_0_7);
            simde_mm_storeu_si128((simde__m128i *)(predictionPtr+3*pStride+8), top_8_15);
            simde_mm_storeu_si128((simde__m128i *)(predictionPtr+3*pStride+16), top_16_23);
            simde_mm_storeu_si128((simde__m128i *)(predictionPtr+3*pStride+24), top_24_31);
            predictionPtr += (pStride << 2);
            simde_mm_storeu_si128((simde__m128i *)(predictionPtr), top_0_7);
            simde_mm_storeu_si128((simde__m128i *)(predictionPtr+8), top_8_15);
            simde_mm_storeu_si128((simde__m128i *)(predictionPtr+16), top_16_23);
            simde_mm_storeu_si128((simde__m128i *)(predictionPtr+24), top_24_31);
            simde_mm_storeu_si128((simde__m128i *)(predictionPtr+pStride), top_0_7);
            simde_mm_storeu_si128((simde__m128i *)(predictionPtr+pStride+8), top_8_15);
            simde_mm_storeu_si128((simde__m128i *)(predictionPtr+pStride+16), top_16_23);
            simde_mm_storeu_si128((simde__m128i *)(predictionPtr+pStride+24), top_24_31);
            simde_mm_storeu_si128((simde__m128i *)(predictionPtr+2*pStride), top_0_7);
            simde_mm_storeu_si128((simde__m128i *)(predictionPtr+2*pStride+8), top_8_15);
            simde_mm_storeu_si128((simde__m128i *)(predictionPtr+2*pStride+16), top_16_23);
            simde_mm_storeu_si128((simde__m128i *)(predictionPtr+2*pStride+24), top_24_31);
            simde_mm_storeu_si128((simde__m128i *)(predictionPtr+3*pStride), top_0_7);
            simde_mm_storeu_si128((simde__m128i *)(predictionPtr+3*pStride+8), top_8_15);
            simde_mm_storeu_si128((simde__m128i *)(predictionPtr+3*pStride+16), top_16_23);
            simde_mm_storeu_si128((simde__m128i *)(predictionPtr+3*pStride+24), top_24_31);
            predictionPtr += (pStride << 2);
        }
        return;
    }

    if (!skip) {
        if (size == 16) {
            simde__m128i top_0_7 = simde_mm_loadu_si128((simde__m128i *)(refSamples + topOffset));      
            simde__m128i top_8_15 = simde_mm_loadu_si128((simde__m128i *)(refSamples + topOffset + 8)); 
            simde_mm_storeu_si128((simde__m128i *)predictionPtr, top_0_7);                         
            simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 8), top_8_15);                  
            simde_mm_storeu_si128((simde__m128i *)(predictionPtr + pStride), top_0_7);         
            simde_mm_storeu_si128((simde__m128i *)(predictionPtr + pStride + 8), top_8_15);    
            simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 2 * pStride), top_0_7);         
            simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 2 * pStride + 8), top_8_15);    
            simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 3 * pStride), top_0_7);        
            simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 3 * pStride + 8), top_8_15);   
            predictionPtr = predictionPtr + (pStride << 2);                              
            simde_mm_storeu_si128((simde__m128i *)predictionPtr, top_0_7);                         
            simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 8), top_8_15);                  
            simde_mm_storeu_si128((simde__m128i *)(predictionPtr + pStride), top_0_7);         
            simde_mm_storeu_si128((simde__m128i *)(predictionPtr + pStride + 8), top_8_15);    
            simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 2 * pStride), top_0_7);         
            simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 2 * pStride + 8), top_8_15);    
            simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 3 * pStride), top_0_7);        
            simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 3 * pStride + 8), top_8_15);   
            predictionPtr = predictionPtr + (pStride << 2);                              
            simde_mm_storeu_si128((simde__m128i *)predictionPtr, top_0_7);                         
            simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 8), top_8_15);                  
            simde_mm_storeu_si128((simde__m128i *)(predictionPtr + pStride), top_0_7);         
            simde_mm_storeu_si128((simde__m128i *)(predictionPtr + pStride + 8), top_8_15);    
            simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 2 * pStride), top_0_7);         
            simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 2 * pStride + 8), top_8_15);    
            simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 3 * pStride), top_0_7);        
            simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 3 * pStride + 8), top_8_15);   
            predictionPtr = predictionPtr + (pStride << 2);                              
            simde_mm_storeu_si128((simde__m128i *)predictionPtr, top_0_7);                         
            simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 8), top_8_15);                  
            simde_mm_storeu_si128((simde__m128i *)(predictionPtr + pStride), top_0_7);         
            simde_mm_storeu_si128((simde__m128i *)(predictionPtr + pStride + 8), top_8_15);    
            simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 2 * pStride), top_0_7);         
            simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 2 * pStride + 8), top_8_15);    
            simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 3 * pStride), top_0_7);        
            simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 3 * pStride + 8), top_8_15);   
        }
        else if (size == 8) {
            simde__m128i top_0_7 = simde_mm_loadu_si128((simde__m128i *)(refSamples + topOffset));    
            simde_mm_storeu_si128((simde__m128i *)predictionPtr, top_0_7);                       
            simde_mm_storeu_si128((simde__m128i *)(predictionPtr + pStride), top_0_7);       
            simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 2 * pStride), top_0_7);       
            simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 3 * pStride), top_0_7);      
            predictionPtr = predictionPtr + (pStride << 2);                            
            simde_mm_storeu_si128((simde__m128i *)(predictionPtr), top_0_7);                     
            simde_mm_storeu_si128((simde__m128i *)(predictionPtr + pStride), top_0_7);       
            simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 2 * pStride), top_0_7);       
            simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 3 * pStride), top_0_7); 
        }
        else {
            simde__m128i top_0_3 = simde_mm_loadl_epi64((simde__m128i *)(refSamples + topOffset));     
            simde_mm_storel_epi64((simde__m128i *)predictionPtr, top_0_3);                        
            simde_mm_storel_epi64((simde__m128i *)(predictionPtr + pStride), top_0_3);        
            simde_mm_storel_epi64((simde__m128i *)(predictionPtr + 2 * pStride), top_0_3);        
            simde_mm_storel_epi64((simde__m128i *)(predictionPtr + 3 * pStride), top_0_3);       
        }
    }
    else {
        pStride <<= 1;
        if (size == 16) {

            simde__m128i top_0_7 = simde_mm_loadu_si128((simde__m128i *)(refSamples+topOffset));
            simde__m128i top_8_15 = simde_mm_loadu_si128((simde__m128i *)(refSamples+topOffset+8));
            simde_mm_storeu_si128((simde__m128i *)(predictionPtr), top_0_7);
            simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 8), top_8_15);
            simde_mm_storeu_si128((simde__m128i *)(predictionPtr + pStride), top_0_7);
            simde_mm_storeu_si128((simde__m128i *)(predictionPtr + pStride + 8), top_8_15);
            simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 2 * pStride), top_0_7);
            simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 2 * pStride + 8), top_8_15);
            simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 3 * pStride), top_0_7);
            simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 3 * pStride + 8), top_8_15);
            predictionPtr += (pStride << 2);
            simde_mm_storeu_si128((simde__m128i *)(predictionPtr), top_0_7);
            simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 8), top_8_15);
            simde_mm_storeu_si128((simde__m128i *)(predictionPtr + pStride), top_0_7);
            simde_mm_storeu_si128((simde__m128i *)(predictionPtr + pStride + 8), top_8_15);
            simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 2 * pStride), top_0_7);
            simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 2 * pStride + 8), top_8_15);
            simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 3 * pStride), top_0_7);
            simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 3 * pStride + 8), top_8_15);
        }
        else if (size == 8) {

            simde__m128i top_0_7 = simde_mm_loadu_si128((simde__m128i *)(refSamples+topOffset));
            simde_mm_storeu_si128((simde__m128i *)(predictionPtr), top_0_7);
            simde_mm_storeu_si128((simde__m128i *)(predictionPtr + pStride), top_0_7);
            simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 2 * pStride), top_0_7);
            simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 3 * pStride), top_0_7);
        }
        else {
            simde__m128i top_0_3 = simde_mm_loadl_epi64((simde__m128i *)(refSamples + topOffset));
            simde_mm_storel_epi64((simde__m128i *)(predictionPtr), top_0_3);
            simde_mm_storel_epi64((simde__m128i *)(predictionPtr+pStride), top_0_3);
        }
    }
    return;
}

void IntraModeHorizontalLuma16bit_SSE2_INTRIN(
    const EB_U32   size,                       //input parameter, denotes the size of the current PU
    EB_U16         *refSamples,                 //input parameter, pointer to the reference samples
    EB_U16         *predictionPtr,              //output parameter, pointer to the prediction
    const EB_U32   predictionBufferStride,     //input parameter, denotes the stride for the prediction ptr
    const EB_BOOL  skip)                       //skip half rows
{    
    EB_U32 pStride = predictionBufferStride;
    EB_U32 leftOffset = 0;
    EB_U32 topOffset = (size << 1) + 1;
    EB_U32 topLeftOffset = size << 1;
    simde__m128i xmm_Max10bit = simde_mm_set1_epi16(0x03FF);
    simde__m128i xmm_0 = simde_mm_setzero_si128();
    
    if (size != 32) {
        simde__m128i topLeft = simde_mm_cvtsi32_si128(*(EB_U32*)(refSamples + topLeftOffset));
        topLeft = simde_mm_unpacklo_epi16(topLeft, topLeft);
        topLeft = simde_mm_unpacklo_epi32(topLeft, topLeft);
        topLeft = simde_mm_unpacklo_epi64(topLeft, topLeft);

        if (!skip) {
            simde__m128i left0_7, left8_15, left0_3, left4_7, left8_11, left12_15, left01, left23, left45, left67, left89, left10_11;
            simde__m128i left12_13, left14_15, left0, left1, left2, left3, left4, left5, left6, left7, left8, left9, left10, left11;
            simde__m128i left12, left13, left14, left15;

            if (size == 16) {
                simde__m128i filter_cmpnt0_7, filter_cmpnt8_15, clip3_0_7, clip3_8_15;

                filter_cmpnt0_7 = simde_mm_srai_epi16(simde_mm_sub_epi16(simde_mm_loadu_si128((simde__m128i *)(refSamples + topOffset)), topLeft), 1);
                filter_cmpnt8_15 = simde_mm_srai_epi16(simde_mm_sub_epi16(simde_mm_loadu_si128((simde__m128i *)(refSamples + topOffset+8)), topLeft), 1);

                left0_7 = simde_mm_loadu_si128((simde__m128i *)(refSamples + leftOffset));
                left8_15 = simde_mm_loadu_si128((simde__m128i *)(refSamples + leftOffset+8));
                
                left0_3 = simde_mm_unpacklo_epi16(left0_7, left0_7);
                left4_7 = simde_mm_unpackhi_epi16(left0_7, left0_7);
                left8_11 = simde_mm_unpacklo_epi16(left8_15, left8_15);
                left12_15 = simde_mm_unpackhi_epi16(left8_15, left8_15);
                
                left01 = simde_mm_unpacklo_epi32(left0_3, left0_3);
                left23 = simde_mm_unpackhi_epi32(left0_3, left0_3);
                left45 = simde_mm_unpacklo_epi32(left4_7, left4_7);
                left67 = simde_mm_unpackhi_epi32(left4_7, left4_7);
                left89 = simde_mm_unpacklo_epi32(left8_11, left8_11);
                left10_11 = simde_mm_unpackhi_epi32(left8_11, left8_11);

                
                left0 = simde_mm_unpacklo_epi64(left01, left01);                
                left1 = simde_mm_unpackhi_epi64(left01, left01);

                clip3_0_7 = simde_mm_max_epi16(simde_mm_min_epi16(simde_mm_add_epi16(filter_cmpnt0_7, left0), xmm_Max10bit), xmm_0);
                clip3_8_15 = simde_mm_max_epi16(simde_mm_min_epi16(simde_mm_add_epi16(filter_cmpnt8_15, left0), xmm_Max10bit), xmm_0);
                
                left14_15 = simde_mm_unpackhi_epi32(left12_15, left12_15);
                left12_13 = simde_mm_unpacklo_epi32(left12_15, left12_15);
                
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr), clip3_0_7);
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 8), clip3_8_15);

                left2 = simde_mm_unpacklo_epi64(left23, left23);
                left3 = simde_mm_unpackhi_epi64(left23, left23);
                left4 = simde_mm_unpacklo_epi64(left45, left45);
                left5 = simde_mm_unpackhi_epi64(left45, left45);
                left6 = simde_mm_unpacklo_epi64(left67, left67);
                left7 = simde_mm_unpackhi_epi64(left67, left67);
                left8 = simde_mm_unpacklo_epi64(left89, left89);
                left9 = simde_mm_unpackhi_epi64(left89, left89);
                left10 = simde_mm_unpacklo_epi64(left10_11, left10_11);
                left11 = simde_mm_unpackhi_epi64(left10_11, left10_11);
                left12 = simde_mm_unpacklo_epi64(left12_13, left12_13);
                left13 = simde_mm_unpackhi_epi64(left12_13, left12_13);
                left14 = simde_mm_unpacklo_epi64(left14_15, left14_15);
                left15 = simde_mm_unpackhi_epi64(left14_15, left14_15);
                
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr+pStride), left1);
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr+pStride+8), left1);
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr+2*pStride), left2);
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr+2*pStride+8), left2);
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr+3*pStride), left3);
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr+3*pStride+8), left3);
                predictionPtr += (pStride << 2);
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr), left4);
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr+8), left4);
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr+pStride), left5);
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr+pStride+8), left5);
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr+2*pStride), left6);
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr+2*pStride+8), left6);
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr+3*pStride), left7);
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr+3*pStride+8), left7);
                predictionPtr += (pStride << 2);
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr), left8);
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr+8), left8);
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr+pStride), left9);
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr+pStride+8), left9);
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr+2*pStride), left10);
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr+2*pStride+8), left10);
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr+3*pStride), left11);
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr+3*pStride+8), left11);
                predictionPtr += (pStride << 2);
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr), left12);
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr+8), left12);
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr+pStride), left13);
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr+pStride+8), left13);
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr+2*pStride), left14);
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr+2*pStride+8), left14);
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr+3*pStride), left15);
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr+3*pStride+8), left15);
            }
            else if (size == 8) {
                simde__m128i filter_cmpnt, clipped_10bit;

                filter_cmpnt = simde_mm_srai_epi16(simde_mm_sub_epi16(simde_mm_loadu_si128((simde__m128i *)(refSamples + topOffset)), topLeft), 1);
                left0_7 = simde_mm_loadu_si128((simde__m128i *)(refSamples + leftOffset));
                left4_7 = simde_mm_unpackhi_epi16(left0_7, left0_7);
                left0_3 = simde_mm_unpacklo_epi16(left0_7, left0_7);
                
                left01 = simde_mm_unpacklo_epi32(left0_3, left0_3);
                left23 = simde_mm_unpackhi_epi32(left0_3, left0_3);
                left45 = simde_mm_unpacklo_epi32(left4_7, left4_7);
                left67 = simde_mm_unpackhi_epi32(left4_7, left4_7);
                
                left0 = simde_mm_unpacklo_epi64(left01, left01);
                left1 = simde_mm_unpackhi_epi64(left01, left01);
                left2 = simde_mm_unpacklo_epi64(left23, left23);
                left3 = simde_mm_unpackhi_epi64(left23, left23);
                left4 = simde_mm_unpacklo_epi64(left45, left45);
                left5 = simde_mm_unpackhi_epi64(left45, left45);
                left6 = simde_mm_unpacklo_epi64(left67, left67);
                left7 = simde_mm_unpackhi_epi64(left67, left67);
                
                clipped_10bit = simde_mm_max_epi16(simde_mm_min_epi16(simde_mm_add_epi16(left0, filter_cmpnt), xmm_Max10bit), xmm_0);

                simde_mm_storeu_si128((simde__m128i *)(predictionPtr), clipped_10bit);
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr+pStride), left1);
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr+2*pStride), left2);
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr+3*pStride), left3);
                predictionPtr += (pStride << 2);
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr), left4);
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr+pStride), left5);
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr+2*pStride), left6);
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr+3*pStride), left7);
            }
            else {
                simde__m128i clipped_10bit, filter_cmpnt;

                filter_cmpnt = simde_mm_srai_epi16(simde_mm_sub_epi16(simde_mm_loadl_epi64((simde__m128i *)(refSamples + topOffset)), topLeft), 1);
                left0_7 = simde_mm_loadl_epi64((simde__m128i *)(refSamples + leftOffset));
                left0_3 = simde_mm_unpacklo_epi16(left0_7, left0_7);

                left23 = simde_mm_unpackhi_epi32(left0_3, left0_3);
                left01 = simde_mm_unpacklo_epi32(left0_3, left0_3);              
                clipped_10bit = simde_mm_max_epi16(simde_mm_min_epi16(simde_mm_add_epi16(left01, filter_cmpnt), xmm_Max10bit), xmm_0);

                simde_mm_storel_epi64((simde__m128i *)(predictionPtr), clipped_10bit);
                simde_mm_storel_epi64((simde__m128i *)(predictionPtr + pStride), simde_mm_srli_si128(left01, 8));
                simde_mm_storel_epi64((simde__m128i *)(predictionPtr + 2 * pStride), left23);
                left23 = simde_mm_srli_si128(left23, 8);
                simde_mm_storel_epi64((simde__m128i *)(predictionPtr + 3 * pStride), left23);
            }
        }
        else {
            pStride <<= 1;

            simde__m128i left0_14_even, left0_6_even, left8_14_even, left02, left46, left8_10, left12_14;
            simde__m128i left0, left2, left4, left6, left8, left10, left12, left14;
            simde__m128i clipped_10bit_0_7, clipped_10bit_8_15, filter_cmpnt_0_7, filter_cmpnt_8_15, skip_mask = simde_mm_set1_epi32(0x0000FFFF);
            if (size == 16) {

                filter_cmpnt_0_7 = simde_mm_srai_epi16(simde_mm_sub_epi16(simde_mm_loadu_si128((simde__m128i *)(refSamples + topOffset)), topLeft), 1);
                filter_cmpnt_8_15 = simde_mm_srai_epi16(simde_mm_sub_epi16(simde_mm_loadu_si128((simde__m128i *)(refSamples + topOffset+8)), topLeft), 1);

                left0_14_even = simde_mm_packs_epi32(simde_mm_and_si128(simde_mm_loadu_si128((simde__m128i *)(refSamples + leftOffset)), skip_mask), 
                                                simde_mm_and_si128(simde_mm_loadu_si128((simde__m128i *)(refSamples + leftOffset + 8)), skip_mask));
                
                left0_6_even = simde_mm_unpacklo_epi16(left0_14_even, left0_14_even);
                left8_14_even = simde_mm_unpackhi_epi16(left0_14_even, left0_14_even);

                left02 = simde_mm_unpacklo_epi32(left0_6_even, left0_6_even);
                left46 = simde_mm_unpackhi_epi32(left0_6_even, left0_6_even);
                left8_10 = simde_mm_unpacklo_epi32(left8_14_even, left8_14_even);
                left12_14 = simde_mm_unpackhi_epi32(left8_14_even, left8_14_even);

                left0 = simde_mm_unpacklo_epi64(left02, left02);
                left2 = simde_mm_unpackhi_epi64(left02, left02);

                clipped_10bit_0_7 = simde_mm_max_epi16(simde_mm_min_epi16(simde_mm_add_epi16(left0, filter_cmpnt_0_7), xmm_Max10bit), xmm_0);
                clipped_10bit_8_15 = simde_mm_max_epi16(simde_mm_min_epi16(simde_mm_add_epi16(left0, filter_cmpnt_8_15), xmm_Max10bit), xmm_0);

                simde_mm_storeu_si128((simde__m128i *)(predictionPtr), clipped_10bit_0_7);
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 8), clipped_10bit_8_15);

                left4 = simde_mm_unpacklo_epi64(left46, left46);
                left6 = simde_mm_unpackhi_epi64(left46, left46);
                left8 = simde_mm_unpacklo_epi64(left8_10, left8_10);
                left10 = simde_mm_unpackhi_epi64(left8_10, left8_10);
                left12 = simde_mm_unpacklo_epi64(left12_14, left12_14);
                left14 = simde_mm_unpackhi_epi64(left12_14, left12_14);
                
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + pStride), left2);
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + pStride + 8), left2);
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 2*pStride), left4);
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 2*pStride + 8), left4);
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 3*pStride), left6);
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 3*pStride + 8), left6);
                predictionPtr += (pStride << 2);
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr), left8);
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 8), left8);
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + pStride), left10);
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + pStride + 8), left10);
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 2*pStride), left12);
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 2*pStride + 8), left12);
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 3*pStride), left14);
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 3*pStride + 8), left14);
            }
            else {

                simde__m128i filter_cmpnt, clipped_10bit;
                simde__m128i left0_14_even, left_0_6_even, left02, left46;
                simde__m128i left0, left2, left4, left6;

                filter_cmpnt = simde_mm_srai_epi16(simde_mm_sub_epi16(simde_mm_loadu_si128((simde__m128i *)(refSamples + topOffset)), topLeft), 1);               

                left0_14_even = simde_mm_packs_epi32(simde_mm_and_si128(simde_mm_loadu_si128((simde__m128i *)(refSamples + leftOffset)), skip_mask), xmm_0);

                left_0_6_even = simde_mm_unpacklo_epi16(left0_14_even, left0_14_even);

                left02 = simde_mm_unpacklo_epi32(left_0_6_even, left_0_6_even);
                left46 = simde_mm_unpackhi_epi32(left_0_6_even, left_0_6_even);

                left0 = simde_mm_unpacklo_epi64(left02, left02);
                left2 = simde_mm_unpackhi_epi64(left02, left02);
                left4 = simde_mm_unpacklo_epi64(left46, left46);
                left6 = simde_mm_unpackhi_epi64(left46, left46);
                                
                clipped_10bit = simde_mm_max_epi16(simde_mm_min_epi16(simde_mm_add_epi16(left0, filter_cmpnt), xmm_Max10bit), xmm_0);

                simde_mm_storeu_si128((simde__m128i *)(predictionPtr), clipped_10bit);
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr+pStride), left2);
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr+2*pStride), left4);
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr+3*pStride), left6);
            }
        }
    }
    else {
        if (!skip) {

            EB_U32 count;
            simde__m128i left0_7, left8_15, left0_3, left4_7, left8_11, left12_15, left01, left23, left45, left67, left89, left10_11;
            simde__m128i left12_13, left14_15, left0, left1, left2, left3, left4, left5, left6, left7, left8, left9, left10, left11;
            simde__m128i left12, left13, left14, left15;
            
            for (count = 0; count < 2; ++count) {
                left0_7 = simde_mm_loadu_si128((simde__m128i *)(refSamples + leftOffset));
                left8_15 = simde_mm_loadu_si128((simde__m128i *)(refSamples + leftOffset + 8));
                refSamples += 16;
                
                left0_3 = simde_mm_unpacklo_epi16(left0_7, left0_7);
                left4_7 = simde_mm_unpackhi_epi16(left0_7, left0_7);
                left8_11 = simde_mm_unpacklo_epi16(left8_15, left8_15);
                left12_15 = simde_mm_unpackhi_epi16(left8_15, left8_15);
                
                left01 = simde_mm_unpacklo_epi32(left0_3, left0_3);
                left23 = simde_mm_unpackhi_epi32(left0_3, left0_3);
                left45 = simde_mm_unpacklo_epi32(left4_7, left4_7);
                left67 = simde_mm_unpackhi_epi32(left4_7, left4_7);
                left89 = simde_mm_unpacklo_epi32(left8_11, left8_11);
                left10_11 = simde_mm_unpackhi_epi32(left8_11, left8_11);
                left12_13 = simde_mm_unpacklo_epi32(left12_15, left12_15);
                left14_15 = simde_mm_unpackhi_epi32(left12_15, left12_15);
                
                left0 = simde_mm_unpacklo_epi64(left01, left01);
                left1 = simde_mm_unpackhi_epi64(left01, left01);
                left2 = simde_mm_unpacklo_epi64(left23, left23);
                left3 = simde_mm_unpackhi_epi64(left23, left23);
                left4 = simde_mm_unpacklo_epi64(left45, left45);
                left5 = simde_mm_unpackhi_epi64(left45, left45);
                left6 = simde_mm_unpacklo_epi64(left67, left67);
                left7 = simde_mm_unpackhi_epi64(left67, left67);
                left8 = simde_mm_unpacklo_epi64(left89, left89);
                left9 = simde_mm_unpackhi_epi64(left89, left89);
                left10 = simde_mm_unpacklo_epi64(left10_11, left10_11);
                left11 = simde_mm_unpackhi_epi64(left10_11, left10_11);
                left12 = simde_mm_unpacklo_epi64(left12_13, left12_13);
                left13 = simde_mm_unpackhi_epi64(left12_13, left12_13);
                left14 = simde_mm_unpacklo_epi64(left14_15, left14_15);
                left15 = simde_mm_unpackhi_epi64(left14_15, left14_15);
                
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr), left0);
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 8), left0);
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 16), left0);
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 24), left0);
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + pStride), left1);
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + pStride + 8), left1);
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + pStride + 16), left1);
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + pStride + 24), left1);
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 2 * pStride), left2);
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 2 * pStride + 8), left2);
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 2 * pStride + 16), left2);
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 2 * pStride + 24), left2);
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 3 * pStride), left3);
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 3 * pStride + 8), left3);
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 3 * pStride + 16), left3);
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 3 * pStride + 24), left3);
                predictionPtr += (pStride << 2);
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr), left4);
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 8), left4);
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 16), left4);
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 24), left4);
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + pStride), left5);
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + pStride + 8), left5);
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + pStride + 16), left5);
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + pStride + 24), left5);
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 2 * pStride), left6);
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 2 * pStride + 8), left6);
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 2 * pStride + 16), left6);
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 2 * pStride + 24), left6);
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 3 * pStride), left7);
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 3 * pStride + 8), left7);
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 3 * pStride + 16), left7);
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 3 * pStride + 24), left7);
                predictionPtr += (pStride << 2);
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr), left8);
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 8), left8);
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 16), left8);
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 24), left8);
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + pStride), left9);
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + pStride + 8), left9);
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + pStride + 16), left9);
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + pStride + 24), left9);
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 2 * pStride), left10);
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 2 * pStride + 8), left10);
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 2 * pStride + 16), left10);
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 2 * pStride + 24), left10);
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 3 * pStride), left11);
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 3 * pStride + 8), left11);
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 3 * pStride + 16), left11);
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 3 * pStride + 24), left11);
                predictionPtr += (pStride << 2);
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr), left12);
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 8), left12);
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 16), left12);
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 24), left12);
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + pStride), left13);
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + pStride + 8), left13);
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + pStride + 16), left13);
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + pStride + 24), left13);
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 2 * pStride), left14);
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 2 * pStride + 8), left14);
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 2 * pStride + 16), left14);
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 2 * pStride + 24), left14);
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 3 * pStride), left15);
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 3 * pStride + 8), left15);
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 3 * pStride + 16), left15);
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 3 * pStride + 24), left15);
                predictionPtr += (pStride << 2);
            }
        }
        else {

            pStride <<= 1;
            simde__m128i left0_14_even, left_0_6_even, left_8_14_even, left02, left46, left8_10, left12_14;
            simde__m128i left0, left2, left4, left6, left8, left10, left12, left14;
            simde__m128i skip_mask = simde_mm_set1_epi32(0x0000FFFF);
            MACRO_HORIZONTAL_LUMA_32X16(0)
            predictionPtr += (pStride << 2);
            MACRO_HORIZONTAL_LUMA_32X16(16)

        }
    }
}

void IntraModeHorizontalChroma16bit_SSE2_INTRIN(
    const EB_U32   size,                       //input parameter, denotes the size of the current PU
    EB_U16         *refSamples,                 //input parameter, pointer to the reference samples
    EB_U16         *predictionPtr,              //output parameter, pointer to the prediction
    const EB_U32   predictionBufferStride,     //input parameter, denotes the stride for the prediction ptr
    const EB_BOOL  skip)                       //skip half rows
{
    EB_U32 pStride = predictionBufferStride;
    EB_U32 leftOffset = 0;

    if (!skip) {
        if (size == 32) {
            EB_U32 count;
            simde__m128i left0_7, left8_15, left0_3, left4_7, left8_11, left12_15, left01, left23, left45, left67, left89, left10_11;
            simde__m128i left12_13, left14_15, left0, left1, left2, left3, left4, left5, left6, left7, left8, left9, left10, left11;
            simde__m128i left12, left13, left14, left15;

            for (count = 0; count < 2; ++count) {
                left0_7 = simde_mm_loadu_si128((simde__m128i *)(refSamples + leftOffset));
                left8_15 = simde_mm_loadu_si128((simde__m128i *)(refSamples + leftOffset + 8));
                refSamples += 16;

                left0_3 = simde_mm_unpacklo_epi16(left0_7, left0_7);
                left4_7 = simde_mm_unpackhi_epi16(left0_7, left0_7);
                left8_11 = simde_mm_unpacklo_epi16(left8_15, left8_15);
                left12_15 = simde_mm_unpackhi_epi16(left8_15, left8_15);

                left01 = simde_mm_unpacklo_epi32(left0_3, left0_3);
                left23 = simde_mm_unpackhi_epi32(left0_3, left0_3);
                left45 = simde_mm_unpacklo_epi32(left4_7, left4_7);
                left67 = simde_mm_unpackhi_epi32(left4_7, left4_7);
                left89 = simde_mm_unpacklo_epi32(left8_11, left8_11);
                left10_11 = simde_mm_unpackhi_epi32(left8_11, left8_11);
                left12_13 = simde_mm_unpacklo_epi32(left12_15, left12_15);
                left14_15 = simde_mm_unpackhi_epi32(left12_15, left12_15);

                left0 = simde_mm_unpacklo_epi64(left01, left01);
                left1 = simde_mm_unpackhi_epi64(left01, left01);
                left2 = simde_mm_unpacklo_epi64(left23, left23);
                left3 = simde_mm_unpackhi_epi64(left23, left23);
                left4 = simde_mm_unpacklo_epi64(left45, left45);
                left5 = simde_mm_unpackhi_epi64(left45, left45);
                left6 = simde_mm_unpacklo_epi64(left67, left67);
                left7 = simde_mm_unpackhi_epi64(left67, left67);
                left8 = simde_mm_unpacklo_epi64(left89, left89);
                left9 = simde_mm_unpackhi_epi64(left89, left89);
                left10 = simde_mm_unpacklo_epi64(left10_11, left10_11);
                left11 = simde_mm_unpackhi_epi64(left10_11, left10_11);
                left12 = simde_mm_unpacklo_epi64(left12_13, left12_13);
                left13 = simde_mm_unpackhi_epi64(left12_13, left12_13);
                left14 = simde_mm_unpacklo_epi64(left14_15, left14_15);
                left15 = simde_mm_unpackhi_epi64(left14_15, left14_15);

                simde_mm_storeu_si128((simde__m128i *)(predictionPtr), left0);
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 8), left0);
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 16), left0);
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 24), left0);
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + pStride), left1);
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + pStride + 8), left1);
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + pStride + 16), left1);
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + pStride + 24), left1);
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 2 * pStride), left2);
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 2 * pStride + 8), left2);
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 2 * pStride + 16), left2);
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 2 * pStride + 24), left2);
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 3 * pStride), left3);
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 3 * pStride + 8), left3);
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 3 * pStride + 16), left3);
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 3 * pStride + 24), left3);
                predictionPtr += (pStride << 2);
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr), left4);
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 8), left4);
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 16), left4);
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 24), left4);
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + pStride), left5);
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + pStride + 8), left5);
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + pStride + 16), left5);
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + pStride + 24), left5);
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 2 * pStride), left6);
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 2 * pStride + 8), left6);
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 2 * pStride + 16), left6);
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 2 * pStride + 24), left6);
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 3 * pStride), left7);
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 3 * pStride + 8), left7);
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 3 * pStride + 16), left7);
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 3 * pStride + 24), left7);
                predictionPtr += (pStride << 2);
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr), left8);
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 8), left8);
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 16), left8);
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 24), left8);
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + pStride), left9);
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + pStride + 8), left9);
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + pStride + 16), left9);
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + pStride + 24), left9);
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 2 * pStride), left10);
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 2 * pStride + 8), left10);
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 2 * pStride + 16), left10);
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 2 * pStride + 24), left10);
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 3 * pStride), left11);
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 3 * pStride + 8), left11);
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 3 * pStride + 16), left11);
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 3 * pStride + 24), left11);
                predictionPtr += (pStride << 2);
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr), left12);
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 8), left12);
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 16), left12);
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 24), left12);
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + pStride), left13);
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + pStride + 8), left13);
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + pStride + 16), left13);
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + pStride + 24), left13);
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 2 * pStride), left14);
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 2 * pStride + 8), left14);
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 2 * pStride + 16), left14);
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 2 * pStride + 24), left14);
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 3 * pStride), left15);
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 3 * pStride + 8), left15);
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 3 * pStride + 16), left15);
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 3 * pStride + 24), left15);
                predictionPtr += (pStride << 2);
            }
        } else if (size == 16) {
            simde__m128i left0_7, left8_15, left0_3, left4_7, left8_11, left12_15;
            simde__m128i left_01, left_23, left_45, left_67, left_89, left_10_11, left_12_13, left_14_15;
            simde__m128i left0, left1, left2, left3, left4, left5, left6, left7, left8, left9, left10, left11, left12, left13, left14, left15;

            left0_7 = simde_mm_loadu_si128((simde__m128i *)(refSamples + leftOffset));
            left8_15 = simde_mm_loadu_si128((simde__m128i *)(refSamples + leftOffset + 8));
            
            left0_3 = simde_mm_unpacklo_epi16(left0_7, left0_7);
            left4_7 = simde_mm_unpackhi_epi16(left0_7, left0_7);
            left8_11 = simde_mm_unpacklo_epi16(left8_15, left8_15);
            left12_15 = simde_mm_unpackhi_epi16(left8_15, left8_15);
            
            left_01 = simde_mm_unpacklo_epi32(left0_3, left0_3);
            left_23 = simde_mm_unpackhi_epi32(left0_3, left0_3);
            left_45 = simde_mm_unpacklo_epi32(left4_7, left4_7);
            left_67 = simde_mm_unpackhi_epi32(left4_7, left4_7);
            left_89 = simde_mm_unpacklo_epi32(left8_11, left8_11);
            left_10_11 = simde_mm_unpackhi_epi32(left8_11, left8_11);
            left_12_13 = simde_mm_unpacklo_epi32(left12_15, left12_15);
            left_14_15 = simde_mm_unpackhi_epi32(left12_15, left12_15);
            
            left0 = simde_mm_unpacklo_epi64(left_01, left_01);
            left1 = simde_mm_unpackhi_epi64(left_01, left_01);
            left2 = simde_mm_unpacklo_epi64(left_23, left_23);
            left3 = simde_mm_unpackhi_epi64(left_23, left_23);
            left4 = simde_mm_unpacklo_epi64(left_45, left_45);
            left5 = simde_mm_unpackhi_epi64(left_45, left_45);
            left6 = simde_mm_unpacklo_epi64(left_67, left_67);
            left7 = simde_mm_unpackhi_epi64(left_67, left_67);
            left8 = simde_mm_unpacklo_epi64(left_89, left_89);
            left9 = simde_mm_unpackhi_epi64(left_89, left_89);
            left10 = simde_mm_unpacklo_epi64(left_10_11, left_10_11);
            left11 = simde_mm_unpackhi_epi64(left_10_11, left_10_11);
            left12 = simde_mm_unpacklo_epi64(left_12_13, left_12_13);
            left13 = simde_mm_unpackhi_epi64(left_12_13, left_12_13);
            left14 = simde_mm_unpacklo_epi64(left_14_15, left_14_15);
            left15 = simde_mm_unpackhi_epi64(left_14_15, left_14_15);
            
            simde_mm_storeu_si128((simde__m128i *)predictionPtr, left0);
            simde_mm_storeu_si128((simde__m128i *)(predictionPtr+8), left0);
            simde_mm_storeu_si128((simde__m128i *)(predictionPtr+pStride), left1);
            simde_mm_storeu_si128((simde__m128i *)(predictionPtr+pStride+8), left1);
            simde_mm_storeu_si128((simde__m128i *)(predictionPtr+2*pStride), left2);
            simde_mm_storeu_si128((simde__m128i *)(predictionPtr+2*pStride+8), left2);
            simde_mm_storeu_si128((simde__m128i *)(predictionPtr+3*pStride), left3);
            simde_mm_storeu_si128((simde__m128i *)(predictionPtr+3*pStride+8), left3);
            predictionPtr += (pStride << 2);
            simde_mm_storeu_si128((simde__m128i *)(predictionPtr), left4);
            simde_mm_storeu_si128((simde__m128i *)(predictionPtr+8), left4);
            simde_mm_storeu_si128((simde__m128i *)(predictionPtr+pStride), left5);
            simde_mm_storeu_si128((simde__m128i *)(predictionPtr+pStride+8), left5);
            simde_mm_storeu_si128((simde__m128i *)(predictionPtr+2*pStride), left6);
            simde_mm_storeu_si128((simde__m128i *)(predictionPtr+2*pStride+8), left6);
            simde_mm_storeu_si128((simde__m128i *)(predictionPtr+3*pStride), left7);
            simde_mm_storeu_si128((simde__m128i *)(predictionPtr+3*pStride+8), left7);
            predictionPtr += (pStride << 2);
            simde_mm_storeu_si128((simde__m128i *)(predictionPtr), left8);
            simde_mm_storeu_si128((simde__m128i *)(predictionPtr+8), left8);
            simde_mm_storeu_si128((simde__m128i *)(predictionPtr+pStride), left9);
            simde_mm_storeu_si128((simde__m128i *)(predictionPtr+pStride+8), left9);
            simde_mm_storeu_si128((simde__m128i *)(predictionPtr+2*pStride), left10);
            simde_mm_storeu_si128((simde__m128i *)(predictionPtr+2*pStride+8), left10);
            simde_mm_storeu_si128((simde__m128i *)(predictionPtr+3*pStride), left11);
            simde_mm_storeu_si128((simde__m128i *)(predictionPtr+3*pStride+8), left11);
            predictionPtr += (pStride << 2);
            simde_mm_storeu_si128((simde__m128i *)(predictionPtr), left12);
            simde_mm_storeu_si128((simde__m128i *)(predictionPtr+8), left12);
            simde_mm_storeu_si128((simde__m128i *)(predictionPtr+pStride), left13);
            simde_mm_storeu_si128((simde__m128i *)(predictionPtr+pStride+8), left13);
            simde_mm_storeu_si128((simde__m128i *)(predictionPtr+2*pStride), left14);
            simde_mm_storeu_si128((simde__m128i *)(predictionPtr+2*pStride+8), left14);
            simde_mm_storeu_si128((simde__m128i *)(predictionPtr+3*pStride), left15);
            simde_mm_storeu_si128((simde__m128i *)(predictionPtr+3*pStride+8), left15);
         }
        else if (size == 8) {
            simde__m128i left0, left1, left2, left3, left4, left5, left6, left7;
            simde__m128i left0_7, left0_3, left4_7, left_01, left_23, left_45, left_67;

            left0_7 = simde_mm_loadu_si128((simde__m128i *)(refSamples + leftOffset));           
            
            left0_3 = simde_mm_unpacklo_epi16(left0_7, left0_7);
            left4_7 = simde_mm_unpackhi_epi16(left0_7, left0_7);
            
            left_01 = simde_mm_unpacklo_epi32(left0_3, left0_3);
            left_23 = simde_mm_unpackhi_epi32(left0_3, left0_3);
            left_45 = simde_mm_unpacklo_epi32(left4_7, left4_7);
            left_67 = simde_mm_unpackhi_epi32(left4_7, left4_7);
            
            left0 = simde_mm_unpacklo_epi64(left_01, left_01);
            left1 = simde_mm_unpackhi_epi64(left_01, left_01);
            left2 = simde_mm_unpacklo_epi64(left_23, left_23);
            left3 = simde_mm_unpackhi_epi64(left_23, left_23);
            left4 = simde_mm_unpacklo_epi64(left_45, left_45);
            left5 = simde_mm_unpackhi_epi64(left_45, left_45);
            left6 = simde_mm_unpacklo_epi64(left_67, left_67);
            left7 = simde_mm_unpackhi_epi64(left_67, left_67);
            
            simde_mm_storeu_si128((simde__m128i *)predictionPtr, left0);
            simde_mm_storeu_si128((simde__m128i *)(predictionPtr + pStride), left1);
            simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 2*pStride), left2);
            simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 3*pStride), left3);
            predictionPtr += (pStride << 2);
            simde_mm_storeu_si128((simde__m128i *)predictionPtr, left4);
            simde_mm_storeu_si128((simde__m128i *)(predictionPtr + pStride), left5);
            simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 2*pStride), left6);
            simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 3*pStride), left7);
        }
        else {
            
            simde__m128i left0_3, left01, left23;
            left0_3 = simde_mm_loadl_epi64((simde__m128i *)(refSamples + leftOffset));
            
            left0_3 = simde_mm_unpacklo_epi16(left0_3, left0_3);

            left01 = simde_mm_unpacklo_epi32(left0_3, left0_3);
            left23 = simde_mm_unpackhi_epi32(left0_3, left0_3);

            simde_mm_storel_epi64((simde__m128i *)predictionPtr, left01);
            simde_mm_storel_epi64((simde__m128i *)(predictionPtr+pStride), simde_mm_srli_si128(left01, 8));
            simde_mm_storel_epi64((simde__m128i *)(predictionPtr+2*pStride), left23);
            simde_mm_storel_epi64((simde__m128i *)(predictionPtr+3*pStride), simde_mm_srli_si128(left23, 8));
        }
    }
    else {
        pStride <<= 1;
        simde__m128i skip_mask = simde_mm_set1_epi32(0x0000FFFF);
        if (size == 32) {
            simde__m128i left0_14_even, left_0_6_even, left_8_14_even, left02, left46, left8_10, left12_14;
            simde__m128i left0, left2, left4, left6, left8, left10, left12, left14;
            MACRO_HORIZONTAL_LUMA_32X16(0)
            predictionPtr += (pStride << 2);
            MACRO_HORIZONTAL_LUMA_32X16(16)
        } else if (size == 16) {

            simde__m128i left0_14_even, left0_6_even, left8_14_even, left02, left46, left8_10, left12_14;
            simde__m128i left0, left2, left4, left6, left8, left10, left12, left14;

            left0_14_even = simde_mm_packs_epi32(simde_mm_and_si128(simde_mm_loadu_si128((simde__m128i *)(refSamples+leftOffset)), skip_mask), 
                                            simde_mm_and_si128(simde_mm_loadu_si128((simde__m128i *)(refSamples+leftOffset+8)), skip_mask));

            left0_6_even = simde_mm_unpacklo_epi16(left0_14_even, left0_14_even);
            left8_14_even = simde_mm_unpackhi_epi16(left0_14_even, left0_14_even);
            
            left02 = simde_mm_unpacklo_epi32(left0_6_even, left0_6_even);
            left46 = simde_mm_unpackhi_epi32(left0_6_even, left0_6_even);
            left8_10 = simde_mm_unpacklo_epi32(left8_14_even, left8_14_even);
            left12_14 = simde_mm_unpackhi_epi32(left8_14_even, left8_14_even);
            
            left0 = simde_mm_unpacklo_epi64(left02, left02);
            left2 = simde_mm_unpackhi_epi64(left02, left02);
            left4 = simde_mm_unpacklo_epi64(left46, left46);
            left6 = simde_mm_unpackhi_epi64(left46, left46);
            left8 = simde_mm_unpacklo_epi64(left8_10, left8_10);
            left10 = simde_mm_unpackhi_epi64(left8_10, left8_10);
            left12 = simde_mm_unpacklo_epi64(left12_14, left12_14);           
            left14 = simde_mm_unpackhi_epi64(left12_14, left12_14);

            simde_mm_storeu_si128((simde__m128i *)(predictionPtr),                 left0);
            simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 8),             left0);
            simde_mm_storeu_si128((simde__m128i *)(predictionPtr + pStride),       left2);
            simde_mm_storeu_si128((simde__m128i *)(predictionPtr + pStride + 8),   left2);
            simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 2*pStride),     left4);
            simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 2*pStride + 8), left4);
            simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 3*pStride),     left6);
            simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 3*pStride + 8), left6);
            predictionPtr += (pStride << 2);
            simde_mm_storeu_si128((simde__m128i *)(predictionPtr),                 left8);
            simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 8),             left8);
            simde_mm_storeu_si128((simde__m128i *)(predictionPtr + pStride),       left10);
            simde_mm_storeu_si128((simde__m128i *)(predictionPtr + pStride + 8),   left10);
            simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 2*pStride),     left12);
            simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 2*pStride + 8), left12);
            simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 3*pStride),     left14);
            simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 3*pStride + 8), left14);
        }
        else if (size == 8) {
            simde__m128i left0_14_even, left0_6_even, left02, left46, left0, left2, left4, left6;

            left0_14_even = simde_mm_and_si128(simde_mm_loadu_si128((simde__m128i *)(refSamples + leftOffset)), skip_mask);
            left0_14_even = simde_mm_packs_epi32(left0_14_even, left0_14_even);    

            left0_6_even = simde_mm_unpacklo_epi16(left0_14_even, left0_14_even);

            left02 = simde_mm_unpacklo_epi32(left0_6_even, left0_6_even);
            left46 = simde_mm_unpackhi_epi32(left0_6_even, left0_6_even);

            left0 = simde_mm_unpacklo_epi64(left02, left02);
            left2 = simde_mm_unpackhi_epi64(left02, left02);
            left4 = simde_mm_unpacklo_epi64(left46, left46);
            left6 = simde_mm_unpackhi_epi64(left46, left46);
            
            simde_mm_storeu_si128((simde__m128i *)(predictionPtr), left0);
            simde_mm_storeu_si128((simde__m128i *)(predictionPtr + pStride), left2);
            simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 2 * pStride), left4);
            simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 3 * pStride), left6);
        }
        else {
            simde__m128i left0_6_even, left02, left0;
            left0_6_even = simde_mm_and_si128(simde_mm_loadl_epi64((simde__m128i *)(refSamples + leftOffset)), skip_mask);
            left0_6_even = simde_mm_packs_epi32(left0_6_even, left0_6_even);

            left02 = simde_mm_unpacklo_epi16(left0_6_even, left0_6_even);
            left0 = simde_mm_unpacklo_epi32(left02, left02);

            simde_mm_storel_epi64((simde__m128i *)predictionPtr, left0);
            simde_mm_storel_epi64((simde__m128i *)(predictionPtr+pStride), simde_mm_srli_si128(left0, 8));
        }     
    }
}

void IntraModeAngular16bit_2_SSE2_INTRIN(
    const EB_U32    size,                       //input parameter, denotes the size of the current PU
    EB_U16         *refSamples,                 //input parameter, pointer to the reference samples
    EB_U16         *predictionPtr,              //output parameter, pointer to the prediction
    const EB_U32    predictionBufferStride,     //input parameter, denotes the stride for the prediction ptr
    const EB_BOOL   skip)
{    
    EB_U32 pStride = predictionBufferStride;
    EB_U32 leftOffset = 0;

    if (!skip) {

        if (size == 32) {
            EB_U32 count;
            
            for (count = 0; count < 8; ++count) {            
                
                simde_mm_storeu_si128((simde__m128i *)predictionPtr,        simde_mm_loadu_si128((simde__m128i *)(refSamples + leftOffset + 1)));
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 8),  simde_mm_loadu_si128((simde__m128i *)(refSamples + leftOffset + 9)));
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 16), simde_mm_loadu_si128((simde__m128i *)(refSamples + leftOffset + 17)));
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 24), simde_mm_loadu_si128((simde__m128i *)(refSamples + leftOffset + 25)));               
                                
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + pStride),      simde_mm_loadu_si128((simde__m128i *)(refSamples + leftOffset + 2)));
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + pStride + 8),  simde_mm_loadu_si128((simde__m128i *)(refSamples + leftOffset + 10)));
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + pStride + 16), simde_mm_loadu_si128((simde__m128i *)(refSamples + leftOffset + 18)));
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + pStride + 24), simde_mm_loadu_si128((simde__m128i *)(refSamples + leftOffset + 26)));              
                
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 2 * pStride),      simde_mm_loadu_si128((simde__m128i *)(refSamples + leftOffset + 3)));
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 2 * pStride + 8),  simde_mm_loadu_si128((simde__m128i *)(refSamples + leftOffset + 11)));
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 2 * pStride + 16), simde_mm_loadu_si128((simde__m128i *)(refSamples + leftOffset + 19)));
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 2 * pStride + 24), simde_mm_loadu_si128((simde__m128i *)(refSamples + leftOffset + 27)));
                
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 3 * pStride),      simde_mm_loadu_si128((simde__m128i *)(refSamples + leftOffset + 4)));
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 3 * pStride + 8),  simde_mm_loadu_si128((simde__m128i *)(refSamples + leftOffset + 12)));
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 3 * pStride + 16), simde_mm_loadu_si128((simde__m128i *)(refSamples + leftOffset + 20)));
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 3 * pStride + 24), simde_mm_loadu_si128((simde__m128i *)(refSamples + leftOffset + 28)));
                refSamples += 4;
                predictionPtr += (pStride << 2);
            }
        }
        else if (size == 16) {
            simde__m128i ref_9, ref_10, ref_11, ref_12, ref_13, ref_14, ref_15, ref_16;
            
            ref_9 = simde_mm_loadu_si128((simde__m128i *)(refSamples + leftOffset + 9));
            ref_10 = simde_mm_loadu_si128((simde__m128i *)(refSamples + leftOffset + 10));
            ref_11 = simde_mm_loadu_si128((simde__m128i *)(refSamples + leftOffset + 11));
            ref_12 = simde_mm_loadu_si128((simde__m128i *)(refSamples + leftOffset + 12));
            ref_13 = simde_mm_loadu_si128((simde__m128i *)(refSamples + leftOffset + 13));
            ref_14 = simde_mm_loadu_si128((simde__m128i *)(refSamples + leftOffset + 14));
            ref_15 = simde_mm_loadu_si128((simde__m128i *)(refSamples + leftOffset + 15));
            ref_16 = simde_mm_loadu_si128((simde__m128i *)(refSamples + leftOffset + 16));

            simde_mm_storeu_si128((simde__m128i *)predictionPtr, simde_mm_loadu_si128((simde__m128i *)(refSamples + leftOffset + 1)));
            simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 8), ref_9);
            
            simde_mm_storeu_si128((simde__m128i *)(predictionPtr + pStride), simde_mm_loadu_si128((simde__m128i *)(refSamples + leftOffset + 2)));
            simde_mm_storeu_si128((simde__m128i *)(predictionPtr + pStride + 8), ref_10);
            
            simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 2 * pStride), simde_mm_loadu_si128((simde__m128i *)(refSamples + leftOffset + 3)));
            simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 2 * pStride + 8), ref_11);

            simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 3 * pStride), simde_mm_loadu_si128((simde__m128i *)(refSamples + leftOffset + 4)));
            simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 3 * pStride + 8), ref_12);
            predictionPtr += (pStride << 2);
            
            simde_mm_storeu_si128((simde__m128i *)predictionPtr, simde_mm_loadu_si128((simde__m128i *)(refSamples + leftOffset + 5)));
            simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 8), ref_13);
            
            simde_mm_storeu_si128((simde__m128i *)(predictionPtr + pStride), simde_mm_loadu_si128((simde__m128i *)(refSamples + leftOffset + 6)));
            simde_mm_storeu_si128((simde__m128i *)(predictionPtr + pStride + 8), ref_14);
            
            simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 2 * pStride), simde_mm_loadu_si128((simde__m128i *)(refSamples + leftOffset + 7)));
            simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 2 * pStride + 8), ref_15);
            
            simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 3 * pStride), simde_mm_loadu_si128((simde__m128i *)(refSamples + leftOffset + 8)));
            simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 3 * pStride + 8), ref_16);
            predictionPtr += (pStride << 2);

            simde_mm_storeu_si128((simde__m128i *)predictionPtr, ref_9);
            simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 8), simde_mm_loadu_si128((simde__m128i *)(refSamples + leftOffset + 17)));

            simde_mm_storeu_si128((simde__m128i *)(predictionPtr + pStride), ref_10);
            simde_mm_storeu_si128((simde__m128i *)(predictionPtr + pStride + 8), simde_mm_loadu_si128((simde__m128i *)(refSamples + leftOffset + 18)));
            
            simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 2 * pStride), ref_11);
            simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 2 * pStride + 8), simde_mm_loadu_si128((simde__m128i *)(refSamples + leftOffset + 19)));
            
            simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 3 * pStride), ref_12);
            simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 3 * pStride + 8), simde_mm_loadu_si128((simde__m128i *)(refSamples + leftOffset + 20)));
            predictionPtr += (pStride << 2);
            
            simde_mm_storeu_si128((simde__m128i *)predictionPtr, ref_13);
            simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 8), simde_mm_loadu_si128((simde__m128i *)(refSamples + leftOffset + 21)));
            
            simde_mm_storeu_si128((simde__m128i *)(predictionPtr + pStride), ref_14);
            simde_mm_storeu_si128((simde__m128i *)(predictionPtr + pStride + 8), simde_mm_loadu_si128((simde__m128i *)(refSamples + leftOffset + 22)));
            
            simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 2 * pStride), ref_15);
            simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 2 * pStride + 8), simde_mm_loadu_si128((simde__m128i *)(refSamples + leftOffset + 23)));
            
            simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 3 * pStride), ref_16);
            simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 3 * pStride + 8), simde_mm_loadu_si128((simde__m128i *)(refSamples + leftOffset + 24)));

        }
        else if (size == 8) {
            
            simde_mm_storeu_si128((simde__m128i *)predictionPtr, simde_mm_loadu_si128((simde__m128i *)(refSamples + leftOffset + 1)));
            simde_mm_storeu_si128((simde__m128i *)(predictionPtr + pStride), simde_mm_loadu_si128((simde__m128i *)(refSamples + leftOffset + 2)));
            simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 2 * pStride), simde_mm_loadu_si128((simde__m128i *)(refSamples + leftOffset + 3)));
            simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 3 * pStride), simde_mm_loadu_si128((simde__m128i *)(refSamples + leftOffset + 4)));
            predictionPtr += (pStride << 2);
            simde_mm_storeu_si128((simde__m128i *)predictionPtr, simde_mm_loadu_si128((simde__m128i *)(refSamples + leftOffset + 5)));
            simde_mm_storeu_si128((simde__m128i *)(predictionPtr + pStride), simde_mm_loadu_si128((simde__m128i *)(refSamples + leftOffset + 6)));
            simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 2 * pStride), simde_mm_loadu_si128((simde__m128i *)(refSamples + leftOffset + 7)));
            simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 3 * pStride), simde_mm_loadu_si128((simde__m128i *)(refSamples + leftOffset + 8)));
        }
        else {
            simde_mm_storel_epi64((simde__m128i *)predictionPtr, simde_mm_loadl_epi64((simde__m128i *)(refSamples + leftOffset + 1)));
            simde_mm_storel_epi64((simde__m128i *)(predictionPtr + pStride), simde_mm_loadl_epi64((simde__m128i *)(refSamples + leftOffset + 2)));
            simde_mm_storel_epi64((simde__m128i *)(predictionPtr + 2 * pStride), simde_mm_loadl_epi64((simde__m128i *)(refSamples + leftOffset + 3)));
            simde_mm_storel_epi64((simde__m128i *)(predictionPtr + 3 * pStride), simde_mm_loadl_epi64((simde__m128i *)(refSamples + leftOffset + 4)));
        }
    }
    else {
        if (size != 4) {
            pStride <<= 1;
            if (size == 32) {
                
                simde__m128i ref9, ref11, ref13, ref15, ref17, ref19, ref21, ref23, ref25, ref27, ref29, ref31, ref33, ref35, ref37, ref39;
                simde__m128i ref41, ref43, ref45, ref47;
                ref9 = simde_mm_loadu_si128((simde__m128i *)(refSamples + leftOffset + 9));
                ref11 = simde_mm_loadu_si128((simde__m128i *)(refSamples + leftOffset + 11));
                ref13 = simde_mm_loadu_si128((simde__m128i *)(refSamples + leftOffset + 13));
                ref15 = simde_mm_loadu_si128((simde__m128i *)(refSamples + leftOffset + 15));
                ref17 = simde_mm_loadu_si128((simde__m128i *)(refSamples + leftOffset + 17));
                ref19 = simde_mm_loadu_si128((simde__m128i *)(refSamples + leftOffset + 19));
                ref21 = simde_mm_loadu_si128((simde__m128i *)(refSamples + leftOffset + 21));
                ref23 = simde_mm_loadu_si128((simde__m128i *)(refSamples + leftOffset + 23));
                ref25 = simde_mm_loadu_si128((simde__m128i *)(refSamples + leftOffset + 25));
                ref27 = simde_mm_loadu_si128((simde__m128i *)(refSamples + leftOffset + 27));
                ref29 = simde_mm_loadu_si128((simde__m128i *)(refSamples + leftOffset + 29));
                ref31 = simde_mm_loadu_si128((simde__m128i *)(refSamples + leftOffset + 31));
                ref33 = simde_mm_loadu_si128((simde__m128i *)(refSamples + leftOffset + 33));
                ref35 = simde_mm_loadu_si128((simde__m128i *)(refSamples + leftOffset + 35));
                ref37 = simde_mm_loadu_si128((simde__m128i *)(refSamples + leftOffset + 37));
                ref39 = simde_mm_loadu_si128((simde__m128i *)(refSamples + leftOffset + 39));
                ref41 = simde_mm_loadu_si128((simde__m128i *)(refSamples + leftOffset + 41));
                ref43 = simde_mm_loadu_si128((simde__m128i *)(refSamples + leftOffset + 43));
                ref45 = simde_mm_loadu_si128((simde__m128i *)(refSamples + leftOffset + 45));
                ref47 = simde_mm_loadu_si128((simde__m128i *)(refSamples + leftOffset + 47));

                simde_mm_storeu_si128((simde__m128i *)predictionPtr, simde_mm_loadu_si128((simde__m128i *)(refSamples + leftOffset + 1)));
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 8), ref9);
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 16), ref17);
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 24), ref25);
                
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + pStride), simde_mm_loadu_si128((simde__m128i *)(refSamples + leftOffset + 3)));
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + pStride + 8), ref11);
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + pStride + 16), ref19);
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + pStride + 24), ref27);             
                
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 2 * pStride), simde_mm_loadu_si128((simde__m128i *)(refSamples + leftOffset + 5)));
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 2 * pStride + 8), ref13);
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 2 * pStride + 16), ref21);
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 2 * pStride + 24), ref29);
                
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 3 * pStride), simde_mm_loadu_si128((simde__m128i *)(refSamples + leftOffset + 7)));
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 3 * pStride + 8), ref15);
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 3 * pStride + 16), ref23);
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 3 * pStride + 24), ref31);
                    
                predictionPtr += (pStride << 2);

                simde_mm_storeu_si128((simde__m128i *)predictionPtr, ref9);
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 8), ref17);
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 16), ref25);
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 24), ref33);
                
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + pStride), ref11);
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + pStride + 8), ref19);
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + pStride + 16), ref27);
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + pStride + 24), ref35);
                
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 2 * pStride), ref13);
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 2 * pStride + 8), ref21);
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 2 * pStride + 16), ref29);
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 2 * pStride + 24), ref37);
                
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 3 * pStride), ref15);
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 3 * pStride + 8), ref23);
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 3 * pStride + 16), ref31);
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 3 * pStride + 24), ref39);
                predictionPtr += (pStride << 2);
                
                simde_mm_storeu_si128((simde__m128i *)predictionPtr, ref17);
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 8), ref25);
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 16), ref33);
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 24), ref41);
                
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + pStride), ref19);
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + pStride + 8), ref27);
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + pStride + 16), ref35);
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + pStride + 24), ref43);
               
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 2 * pStride), ref21);
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 2 * pStride + 8), ref29);
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 2 * pStride + 16), ref37);
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 2 * pStride + 24), ref45);
                
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 3 * pStride), ref23);
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 3 * pStride + 8), ref31);
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 3 * pStride + 16), ref39);
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 3 * pStride + 24), ref47);
                predictionPtr += (pStride << 2);
                
                simde_mm_storeu_si128((simde__m128i *)predictionPtr, ref25);
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 8), ref33);
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 16), ref41);
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 24), simde_mm_loadu_si128((simde__m128i *)(refSamples + leftOffset + 49)));
                
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + pStride), ref27);
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + pStride + 8), ref35);
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + pStride + 16), ref43);
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + pStride + 24), simde_mm_loadu_si128((simde__m128i *)(refSamples + leftOffset + 51)));
                
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 2 * pStride), ref29);
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 2 * pStride + 8), ref37);
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 2 * pStride + 16), ref45);
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 2 * pStride + 24), simde_mm_loadu_si128((simde__m128i *)(refSamples + leftOffset + 53)));
                
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 3 * pStride), ref31);
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 3 * pStride + 8), ref39);
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 3 * pStride + 16), ref47);
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 3 * pStride + 24), simde_mm_loadu_si128((simde__m128i *)(refSamples + leftOffset + 55)));
            }
            else if (size == 16) {
                simde__m128i ref9, ref11, ref13, ref15;

                ref9 = simde_mm_loadu_si128((simde__m128i *)(refSamples + leftOffset + 9));
                ref11 = simde_mm_loadu_si128((simde__m128i *)(refSamples + leftOffset + 11));
                ref13 = simde_mm_loadu_si128((simde__m128i *)(refSamples + leftOffset + 13));
                ref15 = simde_mm_loadu_si128((simde__m128i *)(refSamples + leftOffset + 15));

                simde_mm_storeu_si128((simde__m128i *)predictionPtr, simde_mm_loadu_si128((simde__m128i *)(refSamples + leftOffset + 1)));
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 8), ref9);
                
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + pStride), simde_mm_loadu_si128((simde__m128i *)(refSamples + leftOffset + 3)));
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + pStride + 8), ref11);
                
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 2 * pStride), simde_mm_loadu_si128((simde__m128i *)(refSamples + leftOffset + 5)));
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 2 * pStride + 8), ref13);
                
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 3 * pStride), simde_mm_loadu_si128((simde__m128i *)(refSamples + leftOffset + 7)));
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 3 * pStride + 8), ref15);
                predictionPtr += (pStride << 2);
                
                simde_mm_storeu_si128((simde__m128i *)predictionPtr, ref9);
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 8), simde_mm_loadu_si128((simde__m128i *)(refSamples + leftOffset + 17)));
                
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + pStride), ref11);
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + pStride + 8), simde_mm_loadu_si128((simde__m128i *)(refSamples + leftOffset + 19)));
                
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 2 * pStride), ref13);
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 2 * pStride + 8), simde_mm_loadu_si128((simde__m128i *)(refSamples + leftOffset + 21)));
                
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 3 * pStride), ref15);
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 3 * pStride + 8), simde_mm_loadu_si128((simde__m128i *)(refSamples + leftOffset + 23)));
            }
            else {

                simde_mm_storeu_si128((simde__m128i *)predictionPtr, simde_mm_loadu_si128((simde__m128i *)(refSamples + leftOffset + 1)));
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + pStride), simde_mm_loadu_si128((simde__m128i *)(refSamples + leftOffset + 3)));
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 2 * pStride), simde_mm_loadu_si128((simde__m128i *)(refSamples + leftOffset + 5)));
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 3 * pStride), simde_mm_loadu_si128((simde__m128i *)(refSamples + leftOffset + 7)));
            }
        }
        else {
            simde_mm_storel_epi64((simde__m128i *)(predictionPtr), simde_mm_loadl_epi64((simde__m128i *)(refSamples + leftOffset + 1)));
            simde_mm_storel_epi64((simde__m128i *)(predictionPtr+2*pStride), simde_mm_loadl_epi64((simde__m128i *)(refSamples + leftOffset + 3)));
        }
    }
}

void IntraModeAngular16bit_34_SSE2_INTRIN(
    const EB_U32   size,                       //input parameter, denotes the size of the current PU
    EB_U16         *refSamples,                 //input parameter, pointer to the reference samples
    EB_U16         *predictionPtr,              //output parameter, pointer to the prediction
    const EB_U32   predictionBufferStride,     //input parameter, denotes the stride for the prediction ptr
    const EB_BOOL  skip)
{    
    EB_U32 pStride = predictionBufferStride;
    EB_U32 topOffset = (size << 1) + 1;

    if (!skip) {

        if (size == 32) {
            EB_U32 count;
            
            for (count = 0; count < 8; ++count) {                            
                                
                simde_mm_storeu_si128((simde__m128i *)predictionPtr, simde_mm_loadu_si128((simde__m128i *)(refSamples + topOffset + 1)));
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 8), simde_mm_loadu_si128((simde__m128i *)(refSamples + topOffset + 9)));
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 16), simde_mm_loadu_si128((simde__m128i *)(refSamples + topOffset + 17)));
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 24), simde_mm_loadu_si128((simde__m128i *)(refSamples + topOffset + 25)));
                                
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + pStride), simde_mm_loadu_si128((simde__m128i *)(refSamples + topOffset + 2)));
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + pStride + 8), simde_mm_loadu_si128((simde__m128i *)(refSamples + topOffset + 10)));
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + pStride + 16), simde_mm_loadu_si128((simde__m128i *)(refSamples + topOffset + 18)));
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + pStride + 24), simde_mm_loadu_si128((simde__m128i *)(refSamples + topOffset + 26)));               
                                
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 2 * pStride), simde_mm_loadu_si128((simde__m128i *)(refSamples + topOffset + 3)));
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 2 * pStride + 8), simde_mm_loadu_si128((simde__m128i *)(refSamples + topOffset + 11)));
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 2 * pStride + 16), simde_mm_loadu_si128((simde__m128i *)(refSamples + topOffset + 19)));
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 2 * pStride + 24), simde_mm_loadu_si128((simde__m128i *)(refSamples + topOffset + 27)));
                
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 3 * pStride), simde_mm_loadu_si128((simde__m128i *)(refSamples + topOffset + 4)));
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 3 * pStride + 8), simde_mm_loadu_si128((simde__m128i *)(refSamples + topOffset + 12)));
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 3 * pStride + 16), simde_mm_loadu_si128((simde__m128i *)(refSamples + topOffset + 20)));
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 3 * pStride + 24), simde_mm_loadu_si128((simde__m128i *)(refSamples + topOffset + 28)));
                refSamples += 4;
                predictionPtr += (pStride << 2);
            }
        }
        else if (size == 16) {
            simde__m128i ref_9, ref_10, ref_11, ref_12, ref_13, ref_14, ref_15, ref_16;
            
            ref_9 = simde_mm_loadu_si128((simde__m128i *)(refSamples + topOffset + 9));
            simde_mm_storeu_si128((simde__m128i *)predictionPtr, simde_mm_loadu_si128((simde__m128i *)(refSamples + topOffset + 1)));
            simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 8), ref_9);
            
            ref_10 = simde_mm_loadu_si128((simde__m128i *)(refSamples + topOffset + 10));
            simde_mm_storeu_si128((simde__m128i *)(predictionPtr + pStride), simde_mm_loadu_si128((simde__m128i *)(refSamples + topOffset + 2)));
            simde_mm_storeu_si128((simde__m128i *)(predictionPtr + pStride + 8), ref_10);
            
            ref_11 = simde_mm_loadu_si128((simde__m128i *)(refSamples + topOffset + 11));
            simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 2 * pStride), simde_mm_loadu_si128((simde__m128i *)(refSamples + topOffset + 3)));
            simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 2 * pStride + 8), ref_11);
            
            ref_12 = simde_mm_loadu_si128((simde__m128i *)(refSamples + topOffset + 12));
            simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 3 * pStride), simde_mm_loadu_si128((simde__m128i *)(refSamples + topOffset + 4)));
            simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 3 * pStride + 8), ref_12);
            predictionPtr += (pStride << 2);

            ref_13 = simde_mm_loadu_si128((simde__m128i *)(refSamples + topOffset + 13));
            simde_mm_storeu_si128((simde__m128i *)predictionPtr, simde_mm_loadu_si128((simde__m128i *)(refSamples + topOffset + 5)));
            simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 8), ref_13);

            ref_14 = simde_mm_loadu_si128((simde__m128i *)(refSamples + topOffset + 14));
            simde_mm_storeu_si128((simde__m128i *)(predictionPtr + pStride), simde_mm_loadu_si128((simde__m128i *)(refSamples + topOffset + 6)));
            simde_mm_storeu_si128((simde__m128i *)(predictionPtr + pStride + 8), ref_14);

            ref_15 = simde_mm_loadu_si128((simde__m128i *)(refSamples + topOffset + 15));
            simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 2 * pStride), simde_mm_loadu_si128((simde__m128i *)(refSamples + topOffset + 7)));
            simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 2 * pStride + 8), ref_15);
            
            ref_16 = simde_mm_loadu_si128((simde__m128i *)(refSamples + topOffset + 16));
            simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 3 * pStride), simde_mm_loadu_si128((simde__m128i *)(refSamples + topOffset + 8)));
            simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 3 * pStride + 8), ref_16);
            predictionPtr += (pStride << 2);

            simde_mm_storeu_si128((simde__m128i *)predictionPtr, ref_9);
            simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 8), simde_mm_loadu_si128((simde__m128i *)(refSamples + topOffset + 17)));
            simde_mm_storeu_si128((simde__m128i *)(predictionPtr + pStride), ref_10);
            simde_mm_storeu_si128((simde__m128i *)(predictionPtr + pStride + 8), simde_mm_loadu_si128((simde__m128i *)(refSamples + topOffset + 18)));
            simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 2 * pStride), ref_11);
            simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 2 * pStride + 8), simde_mm_loadu_si128((simde__m128i *)(refSamples + topOffset + 19)));
            simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 3 * pStride), ref_12);
            simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 3 * pStride + 8), simde_mm_loadu_si128((simde__m128i *)(refSamples + topOffset + 20)));
            predictionPtr += (pStride << 2);
            
            simde_mm_storeu_si128((simde__m128i *)predictionPtr, ref_13);
            simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 8), simde_mm_loadu_si128((simde__m128i *)(refSamples + topOffset + 21)));
            simde_mm_storeu_si128((simde__m128i *)(predictionPtr + pStride), ref_14);
            simde_mm_storeu_si128((simde__m128i *)(predictionPtr + pStride + 8), simde_mm_loadu_si128((simde__m128i *)(refSamples + topOffset + 22)));
            simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 2 * pStride), ref_15);
            simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 2 * pStride + 8), simde_mm_loadu_si128((simde__m128i *)(refSamples + topOffset + 23)));
            simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 3 * pStride), ref_16);
            simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 3 * pStride + 8), simde_mm_loadu_si128((simde__m128i *)(refSamples + topOffset + 24)));

        }
        else if (size == 8) {
            
            simde_mm_storeu_si128((simde__m128i *)predictionPtr, simde_mm_loadu_si128((simde__m128i *)(refSamples + topOffset + 1)));
            simde_mm_storeu_si128((simde__m128i *)(predictionPtr + pStride), simde_mm_loadu_si128((simde__m128i *)(refSamples + topOffset + 2)));
            simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 2 * pStride), simde_mm_loadu_si128((simde__m128i *)(refSamples + topOffset + 3)));
            simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 3 * pStride), simde_mm_loadu_si128((simde__m128i *)(refSamples + topOffset + 4)));
            predictionPtr += (pStride << 2);
            simde_mm_storeu_si128((simde__m128i *)predictionPtr, simde_mm_loadu_si128((simde__m128i *)(refSamples + topOffset + 5)));
            simde_mm_storeu_si128((simde__m128i *)(predictionPtr + pStride), simde_mm_loadu_si128((simde__m128i *)(refSamples + topOffset + 6)));
            simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 2 * pStride), simde_mm_loadu_si128((simde__m128i *)(refSamples + topOffset + 7)));
            simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 3 * pStride), simde_mm_loadu_si128((simde__m128i *)(refSamples + topOffset + 8)));
        }
        else {
            simde_mm_storel_epi64((simde__m128i *)predictionPtr, simde_mm_loadl_epi64((simde__m128i *)(refSamples + topOffset + 1)));
            simde_mm_storel_epi64((simde__m128i *)(predictionPtr + pStride), simde_mm_loadl_epi64((simde__m128i *)(refSamples + topOffset + 2)));
            simde_mm_storel_epi64((simde__m128i *)(predictionPtr + 2 * pStride), simde_mm_loadl_epi64((simde__m128i *)(refSamples + topOffset + 3)));
            simde_mm_storel_epi64((simde__m128i *)(predictionPtr + 3 * pStride), simde_mm_loadl_epi64((simde__m128i *)(refSamples + topOffset + 4)));
        }
    }
    else {
        if (size != 4) {
            pStride <<= 1;
            if (size == 32) {
                simde__m128i ref_9, ref_17, ref_25, ref_11, ref_19, ref_27, ref_13, ref_21, ref_29, ref_15, ref_23, ref_31, ref_33, ref_35, ref_37, ref_39;
                simde__m128i ref_41, ref_43, ref_45, ref_47, ref_49, ref_51, ref_53, ref_55;

                ref_9 = simde_mm_loadu_si128((simde__m128i *)(refSamples + topOffset + 9));
                ref_11 = simde_mm_loadu_si128((simde__m128i *)(refSamples + topOffset + 11));
                ref_13 = simde_mm_loadu_si128((simde__m128i *)(refSamples + topOffset + 13));
                ref_15 = simde_mm_loadu_si128((simde__m128i *)(refSamples + topOffset + 15));
                ref_17 = simde_mm_loadu_si128((simde__m128i *)(refSamples + topOffset + 17));
                ref_19 = simde_mm_loadu_si128((simde__m128i *)(refSamples + topOffset + 19));
                ref_21 = simde_mm_loadu_si128((simde__m128i *)(refSamples + topOffset + 21));
                ref_23 = simde_mm_loadu_si128((simde__m128i *)(refSamples + topOffset + 23));
                ref_25 = simde_mm_loadu_si128((simde__m128i *)(refSamples + topOffset + 25));
                ref_27 = simde_mm_loadu_si128((simde__m128i *)(refSamples + topOffset + 27));
                ref_29 = simde_mm_loadu_si128((simde__m128i *)(refSamples + topOffset + 29));
                ref_31 = simde_mm_loadu_si128((simde__m128i *)(refSamples + topOffset + 31));
                ref_33 = simde_mm_loadu_si128((simde__m128i *)(refSamples + topOffset + 33));
                ref_35 = simde_mm_loadu_si128((simde__m128i *)(refSamples + topOffset + 35));
                ref_37 = simde_mm_loadu_si128((simde__m128i *)(refSamples + topOffset + 37));
                ref_39 = simde_mm_loadu_si128((simde__m128i *)(refSamples + topOffset + 39));
                ref_41 = simde_mm_loadu_si128((simde__m128i *)(refSamples + topOffset + 41));
                ref_43 = simde_mm_loadu_si128((simde__m128i *)(refSamples + topOffset + 43));
                ref_45 = simde_mm_loadu_si128((simde__m128i *)(refSamples + topOffset + 45));
                ref_47 = simde_mm_loadu_si128((simde__m128i *)(refSamples + topOffset + 47));
                ref_49 = simde_mm_loadu_si128((simde__m128i *)(refSamples + topOffset + 49));
                ref_51 = simde_mm_loadu_si128((simde__m128i *)(refSamples + topOffset + 51));
                ref_53 = simde_mm_loadu_si128((simde__m128i *)(refSamples + topOffset + 53));
                ref_55 = simde_mm_loadu_si128((simde__m128i *)(refSamples + topOffset + 55));

                simde_mm_storeu_si128((simde__m128i *)predictionPtr, simde_mm_loadu_si128((simde__m128i *)(refSamples + topOffset + 1)));
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 8), ref_9);
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 16), ref_17);
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 24), ref_25);                
                
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + pStride), simde_mm_loadu_si128((simde__m128i *)(refSamples + topOffset + 3)));
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + pStride + 8), ref_11);
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + pStride + 16), ref_19);
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + pStride + 24), ref_27);
                
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 2 * pStride), simde_mm_loadu_si128((simde__m128i *)(refSamples + topOffset + 5)));
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 2 * pStride + 8), ref_13);
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 2 * pStride + 16), ref_21);
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 2 * pStride + 24), ref_29);
                
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 3 * pStride), simde_mm_loadu_si128((simde__m128i *)(refSamples + topOffset + 7)));
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 3 * pStride + 8), ref_15);
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 3 * pStride + 16), ref_23);
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 3 * pStride + 24), ref_31);
                    
                predictionPtr += (pStride << 2);

                simde_mm_storeu_si128((simde__m128i *)predictionPtr,        ref_9);
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 8),  ref_17);
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 16), ref_25);
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 24), ref_33);

                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + pStride),      ref_11);
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + pStride + 8),  ref_19);
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + pStride + 16), ref_27);
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + pStride + 24), ref_35);

                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 2 * pStride),      ref_13);
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 2 * pStride + 8),  ref_21);
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 2 * pStride + 16), ref_29);
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 2 * pStride + 24), ref_37);
                
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 3 * pStride),      ref_15);
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 3 * pStride + 8),  ref_23);
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 3 * pStride + 16), ref_31);
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 3 * pStride + 24), ref_39);

                predictionPtr += (pStride << 2);
               
                simde_mm_storeu_si128((simde__m128i *)predictionPtr,        ref_17);
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 8),  ref_25);
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 16), ref_33);
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 24), ref_41);
                
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + pStride),      ref_19);
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + pStride + 8),  ref_27);
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + pStride + 16), ref_35);
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + pStride + 24), ref_43);
                
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 2 * pStride),      ref_21);
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 2 * pStride + 8),  ref_29);
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 2 * pStride + 16), ref_37);
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 2 * pStride + 24), ref_45);
                
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 3 * pStride),      ref_23);
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 3 * pStride + 8),  ref_31);
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 3 * pStride + 16), ref_39);
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 3 * pStride + 24), ref_47);
                
                predictionPtr += (pStride << 2);
                
                simde_mm_storeu_si128((simde__m128i *)predictionPtr,        ref_25);
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 8),  ref_33);
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 16), ref_41);
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 24), ref_49);
                
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + pStride),      ref_27);
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + pStride + 8),  ref_35);
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + pStride + 16), ref_43);
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + pStride + 24), ref_51);
                
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 2 * pStride),      ref_29);
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 2 * pStride + 8),  ref_37);
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 2 * pStride + 16), ref_45);
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 2 * pStride + 24), ref_53);
                
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 3 * pStride),      ref_31);
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 3 * pStride + 8),  ref_39);
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 3 * pStride + 16), ref_47);
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 3 * pStride + 24), ref_55);
            }
            else if (size == 16) {
                simde__m128i ref_9, ref_11, ref_13, ref_15;

                ref_9 = simde_mm_loadu_si128((simde__m128i *)(refSamples + topOffset + 9));
                ref_11 = simde_mm_loadu_si128((simde__m128i *)(refSamples + topOffset + 11));
                ref_13 = simde_mm_loadu_si128((simde__m128i *)(refSamples + topOffset + 13));
                ref_15 = simde_mm_loadu_si128((simde__m128i *)(refSamples + topOffset + 15));

                simde_mm_storeu_si128((simde__m128i *)predictionPtr, simde_mm_loadu_si128((simde__m128i *)(refSamples + topOffset + 1)));
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 8), ref_9);
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + pStride), simde_mm_loadu_si128((simde__m128i *)(refSamples + topOffset + 3)));
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + pStride + 8), ref_11);
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 2 * pStride), simde_mm_loadu_si128((simde__m128i *)(refSamples + topOffset + 5)));
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 2 * pStride + 8), ref_13);
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 3 * pStride), simde_mm_loadu_si128((simde__m128i *)(refSamples + topOffset + 7)));
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 3 * pStride + 8), ref_15);

                predictionPtr += (pStride << 2);
                
                simde_mm_storeu_si128((simde__m128i *)predictionPtr, ref_9);
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 8), simde_mm_loadu_si128((simde__m128i *)(refSamples + topOffset + 17)));
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + pStride), ref_11);
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + pStride + 8), simde_mm_loadu_si128((simde__m128i *)(refSamples + topOffset + 19)));
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 2 * pStride), ref_13);
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 2 * pStride + 8), simde_mm_loadu_si128((simde__m128i *)(refSamples + topOffset + 21)));
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 3 * pStride), ref_15);
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 3 * pStride + 8), simde_mm_loadu_si128((simde__m128i *)(refSamples + topOffset + 23)));
            }
            else {
                simde_mm_storeu_si128((simde__m128i *)predictionPtr, simde_mm_loadu_si128((simde__m128i *)(refSamples + topOffset + 1)));
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + pStride), simde_mm_loadu_si128((simde__m128i *)(refSamples + topOffset + 3)));
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 2 * pStride), simde_mm_loadu_si128((simde__m128i *)(refSamples + topOffset + 5)));
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 3 * pStride), simde_mm_loadu_si128((simde__m128i *)(refSamples + topOffset + 7)));
            }
        }
        else {
            simde_mm_storel_epi64((simde__m128i *)(predictionPtr), simde_mm_loadl_epi64((simde__m128i *)(refSamples + topOffset + 1)));
            simde_mm_storel_epi64((simde__m128i *)(predictionPtr+2*pStride), simde_mm_loadl_epi64((simde__m128i *)(refSamples + topOffset + 3)));
        }
    }
}

void IntraModeAngular16bit_18_SSE2_INTRIN(
    const EB_U32   size,                       //input parameter, denotes the size of the current PU
    EB_U16         *refSamples,                 //input parameter, pointer to the reference samples
    EB_U16         *predictionPtr,              //output parameter, pointer to the prediction
    const EB_U32   predictionBufferStride,     //input parameter, denotes the stride for the prediction ptr
    const EB_BOOL  skip)    
{    
    EB_U32 pStride = predictionBufferStride;
    EB_U32 topLeftOffset = (size << 1);

    if (!skip) {

        if (size == 32) {
            EB_U32 count;

            for (count = 0; count < 8; ++count) {            
                
                simde_mm_storeu_si128((simde__m128i *)predictionPtr,        simde_mm_loadu_si128((simde__m128i *)(refSamples + topLeftOffset )));
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 8),  simde_mm_loadu_si128((simde__m128i *)(refSamples + topLeftOffset + 8)));
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 16), simde_mm_loadu_si128((simde__m128i *)(refSamples + topLeftOffset + 16)));
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 24), simde_mm_loadu_si128((simde__m128i *)(refSamples + topLeftOffset + 24)));
                
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + pStride),      simde_mm_loadu_si128((simde__m128i *)(refSamples + topLeftOffset - 1)));
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + pStride + 8),  simde_mm_loadu_si128((simde__m128i *)(refSamples + topLeftOffset + 7)));
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + pStride + 16), simde_mm_loadu_si128((simde__m128i *)(refSamples + topLeftOffset + 15)));
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + pStride + 24), simde_mm_loadu_si128((simde__m128i *)(refSamples + topLeftOffset + 23)));
                
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 2 * pStride),      simde_mm_loadu_si128((simde__m128i *)(refSamples + topLeftOffset - 2)));
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 2 * pStride + 8),  simde_mm_loadu_si128((simde__m128i *)(refSamples + topLeftOffset + 6)));
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 2 * pStride + 16), simde_mm_loadu_si128((simde__m128i *)(refSamples + topLeftOffset + 14)));
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 2 * pStride + 24), simde_mm_loadu_si128((simde__m128i *)(refSamples + topLeftOffset + 22)));
                
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 3 * pStride),      simde_mm_loadu_si128((simde__m128i *)(refSamples + topLeftOffset - 3)));
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 3 * pStride + 8),  simde_mm_loadu_si128((simde__m128i *)(refSamples + topLeftOffset + 5)));
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 3 * pStride + 16), simde_mm_loadu_si128((simde__m128i *)(refSamples + topLeftOffset + 13)));
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 3 * pStride + 24), simde_mm_loadu_si128((simde__m128i *)(refSamples + topLeftOffset + 21)));
                refSamples -= 4;
                predictionPtr += (pStride << 2);
            }
        }
        else if (size == 16) {
            simde__m128i xmm_topLeft, xmm_topLeft_n1, xmm_topLeft_n2, xmm_topLeft_n3, xmm_topLeft_n4, xmm_topLeft_n5, xmm_topLeft_n6, xmm_topLeft_n7;
            
            xmm_topLeft = simde_mm_loadu_si128((simde__m128i *)(refSamples + topLeftOffset ));
            xmm_topLeft_n1 = simde_mm_loadu_si128((simde__m128i *)(refSamples + topLeftOffset - 1));
            xmm_topLeft_n2 = simde_mm_loadu_si128((simde__m128i *)(refSamples + topLeftOffset - 2));
            xmm_topLeft_n3 = simde_mm_loadu_si128((simde__m128i *)(refSamples + topLeftOffset - 3));
            xmm_topLeft_n4 = simde_mm_loadu_si128((simde__m128i *)(refSamples + topLeftOffset - 4));
            xmm_topLeft_n5 = simde_mm_loadu_si128((simde__m128i *)(refSamples + topLeftOffset - 5));
            xmm_topLeft_n6 = simde_mm_loadu_si128((simde__m128i *)(refSamples + topLeftOffset - 6));
            xmm_topLeft_n7 = simde_mm_loadu_si128((simde__m128i *)(refSamples + topLeftOffset - 7));

            simde_mm_storeu_si128((simde__m128i *)predictionPtr, xmm_topLeft);
            simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 8), simde_mm_loadu_si128((simde__m128i *)(refSamples + topLeftOffset + 8)));            
            simde_mm_storeu_si128((simde__m128i *)(predictionPtr + pStride), xmm_topLeft_n1);
            simde_mm_storeu_si128((simde__m128i *)(predictionPtr + pStride + 8), simde_mm_loadu_si128((simde__m128i *)(refSamples + topLeftOffset + 7)));
            simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 2 * pStride), xmm_topLeft_n2);
            simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 2 * pStride + 8), simde_mm_loadu_si128((simde__m128i *)(refSamples + topLeftOffset + 6)));            
            simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 3 * pStride), xmm_topLeft_n3);
            simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 3 * pStride + 8), simde_mm_loadu_si128((simde__m128i *)(refSamples + topLeftOffset + 5)));

            predictionPtr += (pStride << 2);

            simde_mm_storeu_si128((simde__m128i *)predictionPtr, xmm_topLeft_n4);
            simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 8), simde_mm_loadu_si128((simde__m128i *)(refSamples + topLeftOffset + 4)));            
            simde_mm_storeu_si128((simde__m128i *)(predictionPtr + pStride), xmm_topLeft_n5);
            simde_mm_storeu_si128((simde__m128i *)(predictionPtr + pStride + 8), simde_mm_loadu_si128((simde__m128i *)(refSamples + topLeftOffset + 3)));
            simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 2 * pStride), xmm_topLeft_n6);
            simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 2 * pStride + 8), simde_mm_loadu_si128((simde__m128i *)(refSamples + topLeftOffset + 2)));
            simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 3 * pStride), xmm_topLeft_n7);
            simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 3 * pStride + 8), simde_mm_loadu_si128((simde__m128i *)(refSamples + topLeftOffset + 1)));

            predictionPtr += (pStride << 2);

            simde_mm_storeu_si128((simde__m128i *)predictionPtr, simde_mm_loadu_si128((simde__m128i *)(refSamples + topLeftOffset - 8)));
            simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 8), xmm_topLeft);
            simde_mm_storeu_si128((simde__m128i *)(predictionPtr + pStride), simde_mm_loadu_si128((simde__m128i *)(refSamples + topLeftOffset - 9)));
            simde_mm_storeu_si128((simde__m128i *)(predictionPtr + pStride + 8), xmm_topLeft_n1);
            simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 2 * pStride), simde_mm_loadu_si128((simde__m128i *)(refSamples + topLeftOffset - 10)));
            simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 2 * pStride + 8), xmm_topLeft_n2);
            simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 3 * pStride), simde_mm_loadu_si128((simde__m128i *)(refSamples + topLeftOffset - 11)));
            simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 3 * pStride + 8), xmm_topLeft_n3);

            predictionPtr += (pStride << 2);
            
            simde_mm_storeu_si128((simde__m128i *)predictionPtr, simde_mm_loadu_si128((simde__m128i *)(refSamples + topLeftOffset - 12)));
            simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 8), xmm_topLeft_n4);
            simde_mm_storeu_si128((simde__m128i *)(predictionPtr + pStride), simde_mm_loadu_si128((simde__m128i *)(refSamples + topLeftOffset - 13)));
            simde_mm_storeu_si128((simde__m128i *)(predictionPtr + pStride + 8), xmm_topLeft_n5);
            simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 2 * pStride), simde_mm_loadu_si128((simde__m128i *)(refSamples + topLeftOffset - 14)));
            simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 2 * pStride + 8), xmm_topLeft_n6);
            simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 3 * pStride), simde_mm_loadu_si128((simde__m128i *)(refSamples + topLeftOffset - 15)));
            simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 3 * pStride + 8), xmm_topLeft_n7);
        }
        else if (size == 8) {
            simde_mm_storeu_si128((simde__m128i *)predictionPtr, simde_mm_loadu_si128((simde__m128i *)(refSamples + topLeftOffset)));
            simde_mm_storeu_si128((simde__m128i *)(predictionPtr + pStride), simde_mm_loadu_si128((simde__m128i *)(refSamples + topLeftOffset - 1)));
            simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 2 * pStride), simde_mm_loadu_si128((simde__m128i *)(refSamples + topLeftOffset - 2)));
            simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 3 * pStride), simde_mm_loadu_si128((simde__m128i *)(refSamples + topLeftOffset - 3)));

            predictionPtr += (pStride << 2);

            simde_mm_storeu_si128((simde__m128i *)predictionPtr, simde_mm_loadu_si128((simde__m128i *)(refSamples + topLeftOffset - 4)));
            simde_mm_storeu_si128((simde__m128i *)(predictionPtr + pStride), simde_mm_loadu_si128((simde__m128i *)(refSamples + topLeftOffset - 5)));
            simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 2 * pStride), simde_mm_loadu_si128((simde__m128i *)(refSamples + topLeftOffset - 6)));
            simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 3 * pStride), simde_mm_loadu_si128((simde__m128i *)(refSamples + topLeftOffset - 7)));
        }
        else {
            simde_mm_storel_epi64((simde__m128i *)predictionPtr, simde_mm_loadl_epi64((simde__m128i *)(refSamples + topLeftOffset)));
            simde_mm_storel_epi64((simde__m128i *)(predictionPtr + pStride), simde_mm_loadl_epi64((simde__m128i *)(refSamples + topLeftOffset - 1)));
            simde_mm_storel_epi64((simde__m128i *)(predictionPtr + 2 * pStride), simde_mm_loadl_epi64((simde__m128i *)(refSamples + topLeftOffset - 2)));
            simde_mm_storel_epi64((simde__m128i *)(predictionPtr + 3 * pStride), simde_mm_loadl_epi64((simde__m128i *)(refSamples + topLeftOffset - 3)));
        }
    }
    else {

        if (size != 4) {
            pStride <<= 1;
            if (size == 32) {

                simde__m128i xmm_topLeft_0, xmm_topLeft_8, xmm_topLeft_16, xmm_topLeft_n8, xmm_topLeft_n2, xmm_topLeft_6, xmm_topLeft_14, xmm_topLeft_n4;
                simde__m128i xmm_topLeft_4, xmm_topLeft_12, xmm_topLeft_n6, xmm_topLeft_2, xmm_topLeft_10, xmm_topLeft_n16, xmm_topLeft_n10, xmm_topLeft_n12, xmm_topLeft_n14;
                simde__m128i xmm_topLeft_n18, xmm_topLeft_n20, xmm_topLeft_n22;

                xmm_topLeft_0 = simde_mm_loadu_si128((simde__m128i *)(refSamples + topLeftOffset ));
                xmm_topLeft_8 = simde_mm_loadu_si128((simde__m128i *)(refSamples + topLeftOffset + 8));
                xmm_topLeft_16 = simde_mm_loadu_si128((simde__m128i *)(refSamples + topLeftOffset + 16));
                xmm_topLeft_n2 = simde_mm_loadu_si128((simde__m128i *)(refSamples + topLeftOffset - 2));
                xmm_topLeft_6 = simde_mm_loadu_si128((simde__m128i *)(refSamples + topLeftOffset + 6));
                xmm_topLeft_14 = simde_mm_loadu_si128((simde__m128i *)(refSamples + topLeftOffset + 14));
                xmm_topLeft_n4 = simde_mm_loadu_si128((simde__m128i *)(refSamples + topLeftOffset - 4));
                xmm_topLeft_4 = simde_mm_loadu_si128((simde__m128i *)(refSamples + topLeftOffset + 4));
                xmm_topLeft_12 = simde_mm_loadu_si128((simde__m128i *)(refSamples + topLeftOffset + 12));
                xmm_topLeft_n6 = simde_mm_loadu_si128((simde__m128i *)(refSamples + topLeftOffset - 6));
                xmm_topLeft_2 = simde_mm_loadu_si128((simde__m128i *)(refSamples + topLeftOffset + 2));
                xmm_topLeft_10 = simde_mm_loadu_si128((simde__m128i *)(refSamples + topLeftOffset + 10));

                xmm_topLeft_n8 = simde_mm_loadu_si128((simde__m128i *)(refSamples + topLeftOffset - 8));
                xmm_topLeft_n10 = simde_mm_loadu_si128((simde__m128i *)(refSamples + topLeftOffset - 10));
                xmm_topLeft_n12 = simde_mm_loadu_si128((simde__m128i *)(refSamples + topLeftOffset - 12));
                xmm_topLeft_n14 = simde_mm_loadu_si128((simde__m128i *)(refSamples + topLeftOffset - 14));
                xmm_topLeft_n16 = simde_mm_loadu_si128((simde__m128i *)(refSamples + topLeftOffset - 16));
                xmm_topLeft_n18 = simde_mm_loadu_si128((simde__m128i *)(refSamples + topLeftOffset - 18));
                xmm_topLeft_n20 = simde_mm_loadu_si128((simde__m128i *)(refSamples + topLeftOffset - 20));
                xmm_topLeft_n22 = simde_mm_loadu_si128((simde__m128i *)(refSamples + topLeftOffset - 22));

                simde_mm_storeu_si128((simde__m128i *)predictionPtr, xmm_topLeft_0);
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 8), xmm_topLeft_8);
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 16), xmm_topLeft_16);
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 24), simde_mm_loadu_si128((simde__m128i *)(refSamples + topLeftOffset + 24))); 
                
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + pStride), xmm_topLeft_n2);
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + pStride + 8), xmm_topLeft_6);
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + pStride + 16), xmm_topLeft_14);
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + pStride + 24), simde_mm_loadu_si128((simde__m128i *)(refSamples + topLeftOffset + 22)));

                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 2 * pStride), xmm_topLeft_n4);
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 2 * pStride + 8), xmm_topLeft_4);
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 2 * pStride + 16), xmm_topLeft_12);
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 2 * pStride + 24), simde_mm_loadu_si128((simde__m128i *)(refSamples + topLeftOffset + 20)));
                
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 3 * pStride), xmm_topLeft_n6);
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 3 * pStride + 8), xmm_topLeft_2);
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 3 * pStride + 16), xmm_topLeft_10);
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 3 * pStride + 24), simde_mm_loadu_si128((simde__m128i *)(refSamples + topLeftOffset + 18)));
                    
                predictionPtr += (pStride << 2);

                simde_mm_storeu_si128((simde__m128i *)predictionPtr, xmm_topLeft_n8);
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 8), xmm_topLeft_0);
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 16), xmm_topLeft_8);
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 24), xmm_topLeft_16);
                
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + pStride), xmm_topLeft_n10);
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + pStride + 8), xmm_topLeft_n2);
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + pStride + 16), xmm_topLeft_6);
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + pStride + 24), xmm_topLeft_14);
                
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 2 * pStride), xmm_topLeft_n12);
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 2 * pStride + 8), xmm_topLeft_n4);
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 2 * pStride + 16), xmm_topLeft_4);
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 2 * pStride + 24), xmm_topLeft_12);

                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 3 * pStride), xmm_topLeft_n14);
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 3 * pStride + 8), xmm_topLeft_n6);
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 3 * pStride + 16), xmm_topLeft_2);
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 3 * pStride + 24), xmm_topLeft_10);
                predictionPtr += (pStride << 2);
                
                simde_mm_storeu_si128((simde__m128i *)predictionPtr, xmm_topLeft_n16);
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 8), xmm_topLeft_n8);
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 16), xmm_topLeft_0);
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 24), xmm_topLeft_8);
                
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + pStride), xmm_topLeft_n18);
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + pStride + 8), xmm_topLeft_n10);
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + pStride + 16), xmm_topLeft_n2);
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + pStride + 24), xmm_topLeft_6);
                
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 2 * pStride), xmm_topLeft_n20);
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 2 * pStride + 8), xmm_topLeft_n12);
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 2 * pStride + 16), xmm_topLeft_n4);
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 2 * pStride + 24), xmm_topLeft_4);
                
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 3 * pStride), xmm_topLeft_n22);
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 3 * pStride + 8), xmm_topLeft_n14);
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 3 * pStride + 16), xmm_topLeft_n6);
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 3 * pStride + 24), xmm_topLeft_2);
                predictionPtr += (pStride << 2);

                simde_mm_storeu_si128((simde__m128i *)predictionPtr, simde_mm_loadu_si128((simde__m128i *)(refSamples + topLeftOffset - 24)));
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 8), xmm_topLeft_n16);
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 16), xmm_topLeft_n8);
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 24), xmm_topLeft_0);
                
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + pStride), simde_mm_loadu_si128((simde__m128i *)(refSamples + topLeftOffset - 26)));
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + pStride + 8), xmm_topLeft_n18);
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + pStride + 16), xmm_topLeft_n10);
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + pStride + 24), xmm_topLeft_n2);
                
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 2 * pStride), simde_mm_loadu_si128((simde__m128i *)(refSamples + topLeftOffset - 28)));
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 2 * pStride + 8), xmm_topLeft_n20);
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 2 * pStride + 16), xmm_topLeft_n12);
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 2 * pStride + 24), xmm_topLeft_n4);
                
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 3 * pStride), simde_mm_loadu_si128((simde__m128i *)(refSamples + topLeftOffset - 30)));
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 3 * pStride + 8), xmm_topLeft_n22);
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 3 * pStride + 16), xmm_topLeft_n14);
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 3 * pStride + 24), xmm_topLeft_n6);
            }
            else if (size == 16) {
                simde__m128i xmm_topLeft_0, xmm_topLeft_n2, xmm_topLeft_n4, xmm_topLeft_n6;

                xmm_topLeft_0 = simde_mm_loadu_si128((simde__m128i *)(refSamples + topLeftOffset ));
                xmm_topLeft_n2 = simde_mm_loadu_si128((simde__m128i *)(refSamples + topLeftOffset - 2));
                xmm_topLeft_n4 = simde_mm_loadu_si128((simde__m128i *)(refSamples + topLeftOffset - 4));
                xmm_topLeft_n6 = simde_mm_loadu_si128((simde__m128i *)(refSamples + topLeftOffset - 6));

                simde_mm_storeu_si128((simde__m128i *)predictionPtr, xmm_topLeft_0);
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 8), simde_mm_loadu_si128((simde__m128i *)(refSamples + topLeftOffset + 8)));
                
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + pStride), xmm_topLeft_n2);
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + pStride + 8), simde_mm_loadu_si128((simde__m128i *)(refSamples + topLeftOffset + 6)));
                
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 2 * pStride), xmm_topLeft_n4);
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 2 * pStride + 8), simde_mm_loadu_si128((simde__m128i *)(refSamples + topLeftOffset + 4)));
                
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 3 * pStride), xmm_topLeft_n6);
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 3 * pStride + 8), simde_mm_loadu_si128((simde__m128i *)(refSamples + topLeftOffset + 2)));

                predictionPtr += (pStride << 2);
                
                simde_mm_storeu_si128((simde__m128i *)predictionPtr, simde_mm_loadu_si128((simde__m128i *)(refSamples + topLeftOffset - 8)));
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 8), xmm_topLeft_0);
                
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + pStride), simde_mm_loadu_si128((simde__m128i *)(refSamples + topLeftOffset - 10)));
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + pStride + 8), xmm_topLeft_n2);
                
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 2 * pStride), simde_mm_loadu_si128((simde__m128i *)(refSamples + topLeftOffset - 12)));
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 2 * pStride + 8), xmm_topLeft_n4);
                
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 3 * pStride), simde_mm_loadu_si128((simde__m128i *)(refSamples + topLeftOffset - 14)));
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 3 * pStride + 8), xmm_topLeft_n6);
            }
            else {
                simde_mm_storeu_si128((simde__m128i *)predictionPtr, simde_mm_loadu_si128((simde__m128i *)(refSamples + topLeftOffset )));
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + pStride), simde_mm_loadu_si128((simde__m128i *)(refSamples + topLeftOffset - 2)));
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 2 * pStride), simde_mm_loadu_si128((simde__m128i *)(refSamples + topLeftOffset - 4)));
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 3 * pStride), simde_mm_loadu_si128((simde__m128i *)(refSamples + topLeftOffset - 6)));
            }
        }
        else {
            simde_mm_storel_epi64((simde__m128i *)(predictionPtr), simde_mm_loadl_epi64((simde__m128i *)(refSamples + topLeftOffset )));
            simde_mm_storel_epi64((simde__m128i *)(predictionPtr+2*pStride), simde_mm_loadl_epi64((simde__m128i *)(refSamples + topLeftOffset - 2)));
        }
    }
}

void IntraModeAngular16bit_Vertical_Kernel_SSE2_INTRIN(
    EB_U32          size,                       //input parameter, denotes the size of the current PU
    EB_U16         *refSampMain,                //input parameter, pointer to the reference samples
    EB_U16         *predictionPtr,              //output parameter, pointer to the prediction
    EB_U32			predictionBufferStride,     //input parameter, denotes the stride for the prediction ptr
    const EB_BOOL	skip,
    EB_S32			intraPredAngle)
{
    simde__m128i xmm_C32 = simde_mm_set1_epi16(0x0020);
    simde__m128i xmm_C16 = simde_mm_srli_epi16(xmm_C32, 1);
    simde__m128i refSampCoeff,  deltaFract, pred_0_7, pred_8_15, pred16_23, pred24_31;
    EB_U32 count, pStride = predictionBufferStride;
    EB_S32 deltaInt, deltaSum = 0;
    refSampMain += 1;

    if (!skip) {
    
        if (size == 32) {

            for (count = 0; count < 32; ++count) {
                deltaSum += intraPredAngle;
                deltaInt = deltaSum >> 5;

                deltaFract = simde_mm_cvtsi32_si128((deltaSum & 31));
                deltaFract = simde_mm_unpacklo_epi16(deltaFract, deltaFract);
                deltaFract = simde_mm_unpacklo_epi32(deltaFract, deltaFract);
                deltaFract = simde_mm_unpacklo_epi64(deltaFract, deltaFract);

                refSampCoeff = simde_mm_sub_epi16(xmm_C32, deltaFract); //(32 - deltaFract)

                //(((32 - deltaFract)*refSampMain[deltaInt + (EB_S32)colIndex] + deltaFract*refSampMain[deltaInt + (EB_S32)colIndex + 1] + 16) >> 5);
                pred_0_7 = simde_mm_srai_epi16(simde_mm_add_epi16(simde_mm_mullo_epi16(simde_mm_loadu_si128((simde__m128i *)(refSampMain + deltaInt)), refSampCoeff), simde_mm_add_epi16(simde_mm_mullo_epi16(simde_mm_loadu_si128((simde__m128i *)(refSampMain + deltaInt + 1)), deltaFract), xmm_C16)), 5);
                pred_8_15 = simde_mm_srai_epi16(simde_mm_add_epi16(simde_mm_mullo_epi16(simde_mm_loadu_si128((simde__m128i *)(refSampMain + deltaInt + 8)), refSampCoeff), simde_mm_add_epi16(simde_mm_mullo_epi16(simde_mm_loadu_si128((simde__m128i *)(refSampMain + deltaInt + 9)), deltaFract), xmm_C16)), 5);
                pred16_23 = simde_mm_srai_epi16(simde_mm_add_epi16(simde_mm_mullo_epi16(simde_mm_loadu_si128((simde__m128i *)(refSampMain + deltaInt + 16)), refSampCoeff), simde_mm_add_epi16(simde_mm_mullo_epi16(simde_mm_loadu_si128((simde__m128i *)(refSampMain + deltaInt + 17)), deltaFract), xmm_C16)), 5);
                pred24_31 = simde_mm_srai_epi16(simde_mm_add_epi16(simde_mm_mullo_epi16(simde_mm_loadu_si128((simde__m128i *)(refSampMain + deltaInt + 24)), refSampCoeff), simde_mm_add_epi16(simde_mm_mullo_epi16(simde_mm_loadu_si128((simde__m128i *)(refSampMain + deltaInt + 25)), deltaFract), xmm_C16)), 5);

                simde_mm_storeu_si128((simde__m128i *)predictionPtr, pred_0_7);
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 8), pred_8_15);
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 16), pred16_23);
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 24), pred24_31);
                predictionPtr += pStride;
            }
        }
        else if (size == 16) {

            for (count = 0; count < 16; ++count) {
                deltaSum += intraPredAngle;
                deltaInt = deltaSum >> 5;

                deltaFract = simde_mm_cvtsi32_si128((deltaSum & 31));
                deltaFract = simde_mm_unpacklo_epi16(deltaFract, deltaFract);
                deltaFract = simde_mm_unpacklo_epi32(deltaFract, deltaFract);
                deltaFract = simde_mm_unpacklo_epi64(deltaFract, deltaFract);

                refSampCoeff = simde_mm_sub_epi16(xmm_C32, deltaFract); //(32 - deltaFract)
                
                //(((32 - deltaFract)*refSampMain[deltaInt + (EB_S32)colIndex] + deltaFract*refSampMain[deltaInt + (EB_S32)colIndex + 1] + 16) >> 5);
                pred_0_7 = simde_mm_srai_epi16(simde_mm_add_epi16(simde_mm_mullo_epi16(simde_mm_loadu_si128((simde__m128i *)(refSampMain + deltaInt)), refSampCoeff), simde_mm_add_epi16(simde_mm_mullo_epi16(simde_mm_loadu_si128((simde__m128i *)(refSampMain + deltaInt + 1)), deltaFract), xmm_C16)), 5);
                pred_8_15 = simde_mm_srai_epi16(simde_mm_add_epi16(simde_mm_mullo_epi16(simde_mm_loadu_si128((simde__m128i *)(refSampMain + deltaInt + 8)), refSampCoeff), simde_mm_add_epi16(simde_mm_mullo_epi16(simde_mm_loadu_si128((simde__m128i *)(refSampMain + deltaInt + 9)), deltaFract), xmm_C16)), 5);
                
                simde_mm_storeu_si128((simde__m128i *)predictionPtr, pred_0_7);
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 8), pred_8_15);
                predictionPtr += pStride;
            }
        }
        else if (size == 8) {

            for (count = 0; count < 8; ++count) {
                deltaSum += intraPredAngle;
                deltaInt = deltaSum >> 5;

                deltaFract = simde_mm_cvtsi32_si128((deltaSum & 31));
                deltaFract = simde_mm_unpacklo_epi16(deltaFract, deltaFract);
                deltaFract = simde_mm_unpacklo_epi32(deltaFract, deltaFract);
                deltaFract = simde_mm_unpacklo_epi64(deltaFract, deltaFract);

                refSampCoeff = simde_mm_sub_epi16(xmm_C32, deltaFract);

                pred_0_7 = simde_mm_srai_epi16(simde_mm_add_epi16(simde_mm_mullo_epi16(simde_mm_loadu_si128((simde__m128i *)(refSampMain + deltaInt)), refSampCoeff), simde_mm_add_epi16(simde_mm_mullo_epi16(simde_mm_loadu_si128((simde__m128i *)(refSampMain + deltaInt + 1)), deltaFract), xmm_C16)), 5);
                
                simde_mm_storeu_si128((simde__m128i *)predictionPtr, pred_0_7);
                predictionPtr += pStride;
            }
        }
        else {
            simde__m128i pred_0_4;

            for (count = 0; count < 4; ++count){
                deltaSum += intraPredAngle;
                deltaInt = deltaSum >> 5;
                
                deltaFract = simde_mm_cvtsi32_si128((deltaSum & 31));
                deltaFract = simde_mm_unpacklo_epi16(deltaFract, deltaFract);
                deltaFract = simde_mm_unpacklo_epi32(deltaFract, deltaFract);

                refSampCoeff = simde_mm_sub_epi16(xmm_C32, deltaFract);
                
                pred_0_4 = simde_mm_srai_epi16(simde_mm_add_epi16(simde_mm_mullo_epi16(simde_mm_loadl_epi64((simde__m128i *)(refSampMain + deltaInt)), refSampCoeff), simde_mm_add_epi16(simde_mm_mullo_epi16(simde_mm_loadl_epi64((simde__m128i *)(refSampMain + deltaInt + 1)), deltaFract), xmm_C16)), 5);
                
                simde_mm_storel_epi64((simde__m128i *)predictionPtr, pred_0_4);
                predictionPtr += pStride;
            }
        }
    }
    else {
        deltaSum = intraPredAngle;
        intraPredAngle <<= 1;

        if (size == 32) {
        
            for (count = 0; count < 16; ++count) {
                deltaInt = deltaSum >> 5;

                deltaFract = simde_mm_cvtsi32_si128((deltaSum & 31));
                deltaFract = simde_mm_unpacklo_epi16(deltaFract, deltaFract);
                deltaFract = simde_mm_unpacklo_epi32(deltaFract, deltaFract);
                deltaFract = simde_mm_unpacklo_epi64(deltaFract, deltaFract);

                refSampCoeff = simde_mm_sub_epi16(xmm_C32, deltaFract);

                pred_0_7 = simde_mm_srai_epi16(simde_mm_add_epi16(simde_mm_mullo_epi16(simde_mm_loadu_si128((simde__m128i *)(refSampMain + deltaInt)), refSampCoeff),      simde_mm_add_epi16(simde_mm_mullo_epi16(simde_mm_loadu_si128((simde__m128i *)(refSampMain + deltaInt + 1)), deltaFract), xmm_C16)), 5);
                pred_8_15 = simde_mm_srai_epi16(simde_mm_add_epi16(simde_mm_mullo_epi16(simde_mm_loadu_si128((simde__m128i *)(refSampMain + deltaInt + 8)), refSampCoeff),  simde_mm_add_epi16(simde_mm_mullo_epi16(simde_mm_loadu_si128((simde__m128i *)(refSampMain + deltaInt + 9)), deltaFract), xmm_C16)), 5);
                pred16_23 = simde_mm_srai_epi16(simde_mm_add_epi16(simde_mm_mullo_epi16(simde_mm_loadu_si128((simde__m128i *)(refSampMain + deltaInt + 16)), refSampCoeff), simde_mm_add_epi16(simde_mm_mullo_epi16(simde_mm_loadu_si128((simde__m128i *)(refSampMain + deltaInt + 17)), deltaFract), xmm_C16)), 5);
                pred24_31 = simde_mm_srai_epi16(simde_mm_add_epi16(simde_mm_mullo_epi16(simde_mm_loadu_si128((simde__m128i *)(refSampMain + deltaInt + 24)), refSampCoeff), simde_mm_add_epi16(simde_mm_mullo_epi16(simde_mm_loadu_si128((simde__m128i *)(refSampMain + deltaInt + 25)), deltaFract), xmm_C16)), 5);

                simde_mm_storeu_si128((simde__m128i *)predictionPtr, pred_0_7);
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 8), pred_8_15);
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 16), pred16_23);
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 24), pred24_31);
                predictionPtr += (pStride << 1);
                deltaSum += intraPredAngle;
            }
        }
        else if (size == 16) {

            for (count = 0; count < 8; ++count) {
                deltaInt = deltaSum >> 5;

                deltaFract = simde_mm_cvtsi32_si128((deltaSum & 31));
                deltaFract = simde_mm_unpacklo_epi16(deltaFract, deltaFract);
                deltaFract = simde_mm_unpacklo_epi32(deltaFract, deltaFract);
                deltaFract = simde_mm_unpacklo_epi64(deltaFract, deltaFract);

                refSampCoeff = simde_mm_sub_epi16(xmm_C32, deltaFract);

                pred_0_7 = simde_mm_srai_epi16(simde_mm_add_epi16(simde_mm_mullo_epi16(simde_mm_loadu_si128((simde__m128i *)(refSampMain + deltaInt)), refSampCoeff), simde_mm_add_epi16(simde_mm_mullo_epi16(simde_mm_loadu_si128((simde__m128i *)(refSampMain + deltaInt + 1)), deltaFract), xmm_C16)), 5);
                pred_8_15 = simde_mm_srai_epi16(simde_mm_add_epi16(simde_mm_mullo_epi16(simde_mm_loadu_si128((simde__m128i *)(refSampMain + deltaInt + 8)), refSampCoeff), simde_mm_add_epi16(simde_mm_mullo_epi16(simde_mm_loadu_si128((simde__m128i *)(refSampMain + deltaInt + 9)), deltaFract), xmm_C16)), 5);
            
                simde_mm_storeu_si128((simde__m128i *)predictionPtr, pred_0_7);
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 8), pred_8_15);
                predictionPtr += (pStride << 1);
                deltaSum += intraPredAngle;
            }
        }
        else if (size == 8) {

            for (count = 0; count < 4; ++count) {
                deltaInt = deltaSum >> 5;
                
                deltaFract = simde_mm_cvtsi32_si128((deltaSum & 31));
                deltaFract = simde_mm_unpacklo_epi16(deltaFract, deltaFract);
                deltaFract = simde_mm_unpacklo_epi32(deltaFract, deltaFract);
                deltaFract = simde_mm_unpacklo_epi64(deltaFract, deltaFract);
                
                refSampCoeff = simde_mm_sub_epi16(xmm_C32, deltaFract);
                
                pred_0_7 = simde_mm_srai_epi16(simde_mm_add_epi16(simde_mm_mullo_epi16(simde_mm_loadu_si128((simde__m128i *)(refSampMain + deltaInt)), refSampCoeff), simde_mm_add_epi16(simde_mm_mullo_epi16(simde_mm_loadu_si128((simde__m128i *)(refSampMain + deltaInt + 1)), deltaFract), xmm_C16)), 5);
                
                simde_mm_storeu_si128((simde__m128i *)predictionPtr, pred_0_7);
                
                predictionPtr += (pStride << 1);
                deltaSum += intraPredAngle;
            }
        }
        else {
            simde__m128i pred_0_4;

            for (count = 0; count < 2; ++count) {
                deltaInt = deltaSum >> 5;
                
                deltaFract = simde_mm_cvtsi32_si128((deltaSum & 31));
                deltaFract = simde_mm_unpacklo_epi16(deltaFract, deltaFract);
                deltaFract = simde_mm_unpacklo_epi32(deltaFract, deltaFract);
                
                refSampCoeff = simde_mm_sub_epi16(xmm_C32, deltaFract);
                
                pred_0_4 = simde_mm_srai_epi16(simde_mm_add_epi16(simde_mm_mullo_epi16(simde_mm_loadl_epi64((simde__m128i *)(refSampMain + deltaInt)), refSampCoeff), simde_mm_add_epi16(simde_mm_mullo_epi16(simde_mm_loadl_epi64((simde__m128i *)(refSampMain + deltaInt+1)), deltaFract), xmm_C16)), 5);
                
                simde_mm_storel_epi64((simde__m128i *)predictionPtr, pred_0_4);
                predictionPtr += (pStride << 1);
                deltaSum += intraPredAngle;
            }
        }
    }
}

void IntraModeAngular16bit_Horizontal_Kernel_SSE2_INTRIN(
    EB_U32         size,                       //input parameter, denotes the size of the current PU
    EB_U16         *refSampMain,                //input parameter, pointer to the reference samples
    EB_U16         *predictionPtr,              //output parameter, pointer to the prediction
    EB_U32         predictionBufferStride,     //input parameter, denotes the stride for the prediction ptr
    const EB_BOOL  skip,
    EB_S32         intraPredAngle)
{
    simde__m128i xmm_C32, xmm_C16, refSampCoeff, deltaFract, skip_mask;
    simde__m128i vertical_temp, vertical_temp0, vertical_temp1, vertical_temp2, vertical_temp3;
    simde__m128i horiz_0, horiz_1, horiz_2, horiz_3, horiz_4, horiz_5, horiz_6, horiz_7, horiz_01h, horiz_23h, horiz_45h, horiz_67h, horiz_0123l, horiz_0123h, horiz_4567l, horiz_4567h;
    EB_U32 count, pStride = predictionBufferStride;
    EB_S32 deltaInt, deltaSum = 0;
    
    xmm_C32 = simde_mm_set1_epi16(0x0020);
    xmm_C16 = simde_mm_srli_epi16(xmm_C32, 1);

    refSampMain += 1;

    if (!skip) {
        
        if (size == 32)
        {        
            EB_S8 temp[0x800];
            EB_U8 outer_counter;
            for (count = 0; count < 256; count += 8) {
                deltaSum += intraPredAngle;
                deltaInt = (deltaSum >> 5);
                
                deltaFract = simde_mm_cvtsi32_si128(deltaSum & 31);
                deltaFract = simde_mm_unpacklo_epi16(deltaFract, deltaFract);
                deltaFract = simde_mm_unpacklo_epi32(deltaFract, deltaFract);
                deltaFract = simde_mm_unpacklo_epi64(deltaFract, deltaFract);
                
                refSampCoeff = simde_mm_sub_epi16(xmm_C32, deltaFract); //(32 - deltaFract)
                
                vertical_temp0 = simde_mm_srai_epi16(simde_mm_add_epi16(simde_mm_mullo_epi16(simde_mm_loadu_si128((simde__m128i *)(refSampMain + deltaInt)), refSampCoeff), simde_mm_add_epi16(simde_mm_mullo_epi16(simde_mm_loadu_si128((simde__m128i *)(refSampMain + deltaInt + 1)), deltaFract), xmm_C16)), 5);
                vertical_temp1 = simde_mm_srai_epi16(simde_mm_add_epi16(simde_mm_mullo_epi16(simde_mm_loadu_si128((simde__m128i *)(refSampMain + deltaInt + 8)), refSampCoeff), simde_mm_add_epi16(simde_mm_mullo_epi16(simde_mm_loadu_si128((simde__m128i *)(refSampMain + deltaInt + 9)), deltaFract), xmm_C16)), 5);
                vertical_temp2 = simde_mm_srai_epi16(simde_mm_add_epi16(simde_mm_mullo_epi16(simde_mm_loadu_si128((simde__m128i *)(refSampMain + deltaInt + 16)), refSampCoeff), simde_mm_add_epi16(simde_mm_mullo_epi16(simde_mm_loadu_si128((simde__m128i *)(refSampMain + deltaInt + 17)), deltaFract), xmm_C16)), 5);
                vertical_temp3 = simde_mm_srai_epi16(simde_mm_add_epi16(simde_mm_mullo_epi16(simde_mm_loadu_si128((simde__m128i *)(refSampMain + deltaInt + 24)), refSampCoeff), simde_mm_add_epi16(simde_mm_mullo_epi16(simde_mm_loadu_si128((simde__m128i *)(refSampMain + deltaInt + 25)), deltaFract), xmm_C16)), 5);
                
                //The order in the result is not horizontal, store in temp buffer and re-arrange later.
                simde_mm_storeu_si128((simde__m128i *)(temp + 8 * count), vertical_temp0);
                simde_mm_storeu_si128((simde__m128i *)(temp + 8 * count + 16), vertical_temp1);
                simde_mm_storeu_si128((simde__m128i *)(temp + 8 * count + 32), vertical_temp2);
                simde_mm_storeu_si128((simde__m128i *)(temp + 8 * count + 48), vertical_temp3);
            }

			EB_S8 * tempPtr = temp;
            EB_U16 * predPtr = predictionPtr;
             
            for (outer_counter = 0; outer_counter < 2; ++outer_counter) {

                for (count = 0; count < 8; ++count) {
                    
                    horiz_0 = simde_mm_unpacklo_epi16(simde_mm_loadl_epi64((simde__m128i *)(tempPtr + 8 * count)), simde_mm_loadl_epi64((simde__m128i *)(tempPtr + 8 * count + 0x040)));
                    horiz_1 = simde_mm_unpacklo_epi16(simde_mm_loadl_epi64((simde__m128i *)(tempPtr + 8 * count + 0x080)), simde_mm_loadl_epi64((simde__m128i *)(tempPtr + 8 * count + 0x0C0)));
                    horiz_2 = simde_mm_unpacklo_epi16(simde_mm_loadl_epi64((simde__m128i *)(tempPtr + 8 * count + 0x100)), simde_mm_loadl_epi64((simde__m128i *)(tempPtr + 8 * count + 0x140)));
                    horiz_3 = simde_mm_unpacklo_epi16(simde_mm_loadl_epi64((simde__m128i *)(tempPtr + 8 * count + 0x180)), simde_mm_loadl_epi64((simde__m128i *)(tempPtr + 8 * count + 0x1C0)));
                    horiz_4 = simde_mm_unpacklo_epi16(simde_mm_loadl_epi64((simde__m128i *)(tempPtr + 8 * count + 0x200)), simde_mm_loadl_epi64((simde__m128i *)(tempPtr + 8 * count + 0x240)));
                    horiz_5 = simde_mm_unpacklo_epi16(simde_mm_loadl_epi64((simde__m128i *)(tempPtr + 8 * count + 0x280)), simde_mm_loadl_epi64((simde__m128i *)(tempPtr + 8 * count + 0x2C0)));
                    horiz_6 = simde_mm_unpacklo_epi16(simde_mm_loadl_epi64((simde__m128i *)(tempPtr + 8 * count + 0x300)), simde_mm_loadl_epi64((simde__m128i *)(tempPtr + 8 * count + 0x340)));
                    horiz_7 = simde_mm_unpacklo_epi16(simde_mm_loadl_epi64((simde__m128i *)(tempPtr + 8 * count + 0x380)), simde_mm_loadl_epi64((simde__m128i *)(tempPtr + 8 * count + 0x3C0)));

                    MACRO_UNPACK(32, horiz_0, horiz_1, horiz_2, horiz_3, horiz_4, horiz_5, horiz_6, horiz_7, horiz_01h, horiz_23h, horiz_45h, horiz_67h)
                    MACRO_UNPACK(64, horiz_0, horiz_2, horiz_01h, horiz_23h, horiz_4, horiz_6, horiz_45h, horiz_67h, horiz_0123l, horiz_0123h, horiz_4567l, horiz_4567h)

                    simde_mm_storeu_si128((simde__m128i *)(predPtr), horiz_0);
                    simde_mm_storeu_si128((simde__m128i *)(predPtr + 8), horiz_4);
                    simde_mm_storeu_si128((simde__m128i *)(predPtr + pStride), horiz_0123l);
                    simde_mm_storeu_si128((simde__m128i *)(predPtr + pStride + 8), horiz_4567l);
                    simde_mm_storeu_si128((simde__m128i *)(predPtr + 2 * pStride), horiz_01h);
                    simde_mm_storeu_si128((simde__m128i *)(predPtr + 2 * pStride + 8), horiz_45h);
                    simde_mm_storeu_si128((simde__m128i *)(predPtr + 3 * pStride), horiz_0123h);
                    simde_mm_storeu_si128((simde__m128i *)(predPtr + 3 * pStride + 8), horiz_4567h);
                    predPtr += 4 * pStride;
                }
                tempPtr = temp + 0x400;
                predPtr = predictionPtr + 16;
            }
        }
        else if (size == 16) {
            
            EB_S8 temp[0x200];

            for (count = 0; count < 64; count+=4) {
                deltaSum += intraPredAngle;
                deltaInt = (deltaSum >> 5);
                
                deltaFract = simde_mm_cvtsi32_si128(deltaSum & 31);
                deltaFract = simde_mm_unpacklo_epi16(deltaFract, deltaFract);
                deltaFract = simde_mm_unpacklo_epi32(deltaFract, deltaFract);
                deltaFract = simde_mm_unpacklo_epi64(deltaFract, deltaFract);
                
                refSampCoeff = simde_mm_sub_epi16(xmm_C32, deltaFract);
                
                vertical_temp0 = simde_mm_srai_epi16(simde_mm_add_epi16(simde_mm_mullo_epi16(simde_mm_loadu_si128((simde__m128i *)(refSampMain + deltaInt)), refSampCoeff), simde_mm_add_epi16(simde_mm_mullo_epi16(simde_mm_loadu_si128((simde__m128i *)(refSampMain + deltaInt + 1)), deltaFract), xmm_C16)), 5);
                vertical_temp1 = simde_mm_srai_epi16(simde_mm_add_epi16(simde_mm_mullo_epi16(simde_mm_loadu_si128((simde__m128i *)(refSampMain + deltaInt + 8)), refSampCoeff), simde_mm_add_epi16(simde_mm_mullo_epi16(simde_mm_loadu_si128((simde__m128i *)(refSampMain + deltaInt + 9)), deltaFract), xmm_C16)), 5);
                
                //The order in the result is not horizontal, store in temp buffer and re-arrange later.
                simde_mm_storeu_si128((simde__m128i *)(temp + 8 * count), vertical_temp0);
                simde_mm_storeu_si128((simde__m128i *)(temp + 8 * count + 16), vertical_temp1);
            }

            for (count = 0; count < 4; ++count) {

                horiz_0 = simde_mm_unpacklo_epi16(simde_mm_loadl_epi64((simde__m128i *)(temp + 8 * count )), simde_mm_loadl_epi64((simde__m128i *)(temp + 8 * count + 0x020)));
                horiz_1 = simde_mm_unpacklo_epi16(simde_mm_loadl_epi64((simde__m128i *)(temp + 8 * count + 0x040)), simde_mm_loadl_epi64((simde__m128i *)(temp + 8 * count + 0x060)));
                horiz_2 = simde_mm_unpacklo_epi16(simde_mm_loadl_epi64((simde__m128i *)(temp + 8 * count + 0x080)), simde_mm_loadl_epi64((simde__m128i *)(temp + 8 * count + 0x0A0)));
                horiz_3 = simde_mm_unpacklo_epi16(simde_mm_loadl_epi64((simde__m128i *)(temp + 8 * count + 0x0C0)), simde_mm_loadl_epi64((simde__m128i *)(temp + 8 * count + 0x0E0)));
                horiz_4 = simde_mm_unpacklo_epi16(simde_mm_loadl_epi64((simde__m128i *)(temp + 8 * count + 0x100)), simde_mm_loadl_epi64((simde__m128i *)(temp + 8 * count + 0x120)));
                horiz_5 = simde_mm_unpacklo_epi16(simde_mm_loadl_epi64((simde__m128i *)(temp + 8 * count + 0x140)), simde_mm_loadl_epi64((simde__m128i *)(temp + 8 * count + 0x160)));
                horiz_6 = simde_mm_unpacklo_epi16(simde_mm_loadl_epi64((simde__m128i *)(temp + 8 * count + 0x180)), simde_mm_loadl_epi64((simde__m128i *)(temp + 8 * count + 0x1A0)));
                horiz_7 = simde_mm_unpacklo_epi16(simde_mm_loadl_epi64((simde__m128i *)(temp + 8 * count + 0x1C0)), simde_mm_loadl_epi64((simde__m128i *)(temp + 8 * count + 0x1E0)));

                MACRO_UNPACK(32, horiz_0, horiz_1, horiz_2, horiz_3, horiz_4, horiz_5, horiz_6, horiz_7, horiz_01h, horiz_23h, horiz_45h, horiz_67h)
                MACRO_UNPACK(64, horiz_0, horiz_2, horiz_01h, horiz_23h, horiz_4, horiz_6, horiz_45h, horiz_67h, horiz_0123l, horiz_0123h, horiz_4567l, horiz_4567h)

                simde_mm_storeu_si128((simde__m128i *)predictionPtr, horiz_0);
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 8), horiz_4);
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + pStride), horiz_0123l);
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + pStride + 8), horiz_4567l);
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 2 * pStride), horiz_01h);
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 2 * pStride + 8), horiz_45h);
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 3 * pStride), horiz_0123h);
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 3 * pStride + 8), horiz_4567h);
                predictionPtr += 4 * pStride;
            }
        }
        else if (size == 8) {
            simde__m128i horiz04h, horiz07l, horiz0145, horiz07h;
            simde__m128i horiz_0, horiz_1, horiz_2, horiz_3, horiz_4, horiz_5, horiz_6, horiz_7, horiz_01h, horiz_23h, horiz_45h, horiz_67h, horiz_0123l, horiz_0123h, horiz_4567l, horiz_4567h;
            EB_S8 temp[128];
            for (count = 0; count < 16; count+=2) {
                deltaSum += intraPredAngle;
                deltaInt = (deltaSum >> 5);
                
                deltaFract = simde_mm_cvtsi32_si128(deltaSum & 31);
                deltaFract = simde_mm_unpacklo_epi16(deltaFract, deltaFract);
                deltaFract = simde_mm_unpacklo_epi32(deltaFract, deltaFract);
                deltaFract = simde_mm_unpacklo_epi64(deltaFract, deltaFract);
                
                refSampCoeff = simde_mm_sub_epi16(xmm_C32, deltaFract);
                
                vertical_temp = simde_mm_srai_epi16(simde_mm_add_epi16(simde_mm_mullo_epi16(simde_mm_loadu_si128((simde__m128i *)(refSampMain + deltaInt)), refSampCoeff), simde_mm_add_epi16(simde_mm_mullo_epi16(simde_mm_loadu_si128((simde__m128i *)(refSampMain + deltaInt + 1)), deltaFract), xmm_C16)), 5);
                
                //The order in the result is not horizontal, store in temp buffer and re-arrange later.
                simde_mm_storeu_si128((simde__m128i *)(temp + 8 * count), vertical_temp);
            }
            horiz_0 = simde_mm_loadu_si128((simde__m128i *)(temp));
            horiz_1 = simde_mm_loadu_si128((simde__m128i *)(temp + 0x10));
            horiz_2 = simde_mm_loadu_si128((simde__m128i *)(temp + 0x20));
            horiz_3 = simde_mm_loadu_si128((simde__m128i *)(temp + 0x30));
            horiz_4 = simde_mm_loadu_si128((simde__m128i *)(temp + 0x40));
            horiz_5 = simde_mm_loadu_si128((simde__m128i *)(temp + 0x50));
            horiz_6 = simde_mm_loadu_si128((simde__m128i *)(temp + 0x60));
            horiz_7 = simde_mm_loadu_si128((simde__m128i *)(temp + 0x70));
            
            //Unpack to re-arrange the packing to the horizontal packing
            MACRO_UNPACK(16, horiz_0, horiz_1, horiz_2, horiz_3, horiz_4, horiz_5, horiz_6, horiz_7, horiz_01h, horiz_23h, horiz_45h, horiz_67h)
            MACRO_UNPACK(32, horiz_0, horiz_2, horiz_01h, horiz_23h, horiz_4, horiz_6, horiz_45h, horiz_67h, horiz_0123l, horiz_0123h, horiz_4567l, horiz_4567h)
            MACRO_UNPACK(64, horiz_0, horiz_4, horiz_0123l, horiz_4567l, horiz_01h, horiz_45h, horiz_0123h, horiz_4567h, horiz04h, horiz07l, horiz0145, horiz07h)

            simde_mm_storeu_si128((simde__m128i *)predictionPtr, horiz_0);
            simde_mm_storeu_si128((simde__m128i *)(predictionPtr + pStride), horiz04h);
            simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 2 * pStride), horiz_0123l);
            simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 3 * pStride), horiz07l);
            predictionPtr += 4 * pStride;
            simde_mm_storeu_si128((simde__m128i *)predictionPtr, horiz_01h);
            simde_mm_storeu_si128((simde__m128i *)(predictionPtr + pStride), horiz0145);
            simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 2 * pStride), horiz_0123h);
            simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 3 * pStride), horiz07h);
        }
        else {
             EB_S64 temp[4];
             simde__m128i horiz_0, horiz_1, pred_lo, pred_hi;

            for (count = 0; count < 4; ++count) {
                deltaSum += intraPredAngle;
                deltaInt = (deltaSum >> 5);
                
                deltaFract = simde_mm_cvtsi32_si128(deltaSum & 31);
                deltaFract = simde_mm_unpacklo_epi16(deltaFract, deltaFract);
                deltaFract = simde_mm_unpacklo_epi32(deltaFract, deltaFract);
                
                refSampCoeff = simde_mm_sub_epi16(xmm_C32, deltaFract);
                
                vertical_temp = simde_mm_srai_epi16(simde_mm_add_epi16(simde_mm_mullo_epi16(simde_mm_loadl_epi64((simde__m128i *)(refSampMain + deltaInt)), refSampCoeff), simde_mm_add_epi16(simde_mm_mullo_epi16(simde_mm_loadl_epi64((simde__m128i *)(refSampMain + deltaInt + 1)), deltaFract), xmm_C16)), 5);
                
                //The order in the result is not horizontal, store in temp buffer and re-arrange later.
                temp[count] = simde_mm_cvtsi128_si64(vertical_temp);
            }
            //Unpack to re-arrange the packing to the horizontal packing
            horiz_0 = simde_mm_unpacklo_epi16(simde_mm_cvtsi64_si128(temp[0]), simde_mm_cvtsi64_si128(temp[1]));
            horiz_1 = simde_mm_unpacklo_epi16(simde_mm_cvtsi64_si128(temp[2]), simde_mm_cvtsi64_si128(temp[3]));
            
            pred_hi = simde_mm_unpackhi_epi32(horiz_0, horiz_1);
            pred_lo = simde_mm_unpacklo_epi32(horiz_0, horiz_1);
            
            simde_mm_storel_epi64((simde__m128i *)predictionPtr, pred_lo);
            simde_mm_storel_epi64((simde__m128i *)(predictionPtr + pStride), simde_mm_srli_si128(pred_lo, 8));
            simde_mm_storel_epi64((simde__m128i *)(predictionPtr + 2 * pStride), pred_hi);
            simde_mm_storel_epi64((simde__m128i *)(predictionPtr + 3 * pStride), simde_mm_srli_si128(pred_hi, 8));
        }
    }
    else {
        pStride <<= 1;
        skip_mask = simde_mm_set1_epi32(0x0000FFFF);
        if (size == 32) {

            EB_S8 temp[0x400];
            simde__m128i refSamp_0_15, refSamp_1_16, refSamp16_31, refSamp17_32;

            for (count = 0; count < 128; count += 4) {
                deltaSum += intraPredAngle;
                deltaInt = (deltaSum >> 5);
                
                refSamp_0_15 = simde_mm_packs_epi32(simde_mm_and_si128(simde_mm_loadu_si128((simde__m128i *)(refSampMain + deltaInt)), skip_mask), simde_mm_and_si128(simde_mm_loadu_si128((simde__m128i *)(refSampMain + deltaInt + 8)), skip_mask));
                refSamp_1_16 = simde_mm_packs_epi32(simde_mm_and_si128(simde_mm_loadu_si128((simde__m128i *)(refSampMain + deltaInt + 1)), skip_mask), simde_mm_and_si128(simde_mm_loadu_si128((simde__m128i *)(refSampMain + deltaInt + 9)), skip_mask));
                refSamp16_31 = simde_mm_packs_epi32(simde_mm_and_si128(simde_mm_loadu_si128((simde__m128i *)(refSampMain + deltaInt + 16)), skip_mask), simde_mm_and_si128(simde_mm_loadu_si128((simde__m128i *)(refSampMain + deltaInt + 24)), skip_mask));
                refSamp17_32 = simde_mm_packs_epi32(simde_mm_and_si128(simde_mm_loadu_si128((simde__m128i *)(refSampMain + deltaInt + 17)), skip_mask), simde_mm_and_si128(simde_mm_loadu_si128((simde__m128i *)(refSampMain + deltaInt + 25)), skip_mask));

                deltaFract = simde_mm_cvtsi32_si128(deltaSum & 31);
                deltaFract = simde_mm_unpacklo_epi16(deltaFract, deltaFract);
                deltaFract = simde_mm_unpacklo_epi32(deltaFract, deltaFract);
                deltaFract = simde_mm_unpacklo_epi64(deltaFract, deltaFract);
                
                refSampCoeff = simde_mm_sub_epi16(xmm_C32, deltaFract);
                
                //The order in the result is not horizontal, store in temp buffer and re-arrange later.
                vertical_temp0 = simde_mm_srai_epi16(simde_mm_add_epi16(simde_mm_mullo_epi16(refSamp_0_15, refSampCoeff), simde_mm_add_epi16(simde_mm_mullo_epi16(refSamp_1_16, deltaFract), xmm_C16)), 5);
                vertical_temp1 = simde_mm_srai_epi16(simde_mm_add_epi16(simde_mm_mullo_epi16(refSamp16_31, refSampCoeff), simde_mm_add_epi16(simde_mm_mullo_epi16(refSamp17_32, deltaFract), xmm_C16)), 5);

                simde_mm_storeu_si128((simde__m128i *)(temp + 8 * count), vertical_temp0);
                simde_mm_storeu_si128((simde__m128i *)(temp + 8 * count + 16), vertical_temp1);
            }
			EB_S8 * tempPtr = temp;
            EB_U16 * predPtr = predictionPtr;
            EB_U8 outer_counter;
            simde__m128i horiz_0, horiz_1, horiz_2, horiz_3, horiz_4, horiz_5, horiz_6, horiz_7, horiz_01h, horiz_23h, horiz_45h, horiz_67h, horiz_0123l, horiz_0123h, horiz_4567l, horiz_4567h;;

            for (outer_counter = 0; outer_counter < 2; ++outer_counter) {
                for (count = 0; count < 4; ++count) {

                    horiz_0 = simde_mm_unpacklo_epi16(simde_mm_loadl_epi64((simde__m128i *)(tempPtr + 8 * count)), simde_mm_loadl_epi64((simde__m128i *)(tempPtr + 8 * count + 0x020)));
                    horiz_1 = simde_mm_unpacklo_epi16(simde_mm_loadl_epi64((simde__m128i *)(tempPtr + 8 * count + 0x040)), simde_mm_loadl_epi64((simde__m128i *)(tempPtr + 8 * count + 0x060)));
                    horiz_2 = simde_mm_unpacklo_epi16(simde_mm_loadl_epi64((simde__m128i *)(tempPtr + 8 * count + 0x080)), simde_mm_loadl_epi64((simde__m128i *)(tempPtr + 8 * count + 0x0A0)));
                    horiz_3 = simde_mm_unpacklo_epi16(simde_mm_loadl_epi64((simde__m128i *)(tempPtr + 8 * count + 0x0C0)), simde_mm_loadl_epi64((simde__m128i *)(tempPtr + 8 * count + 0x0E0)));
                    horiz_4 = simde_mm_unpacklo_epi16(simde_mm_loadl_epi64((simde__m128i *)(tempPtr + 8 * count + 0x100)), simde_mm_loadl_epi64((simde__m128i *)(tempPtr + 8 * count + 0x120)));
                    horiz_5 = simde_mm_unpacklo_epi16(simde_mm_loadl_epi64((simde__m128i *)(tempPtr + 8 * count + 0x140)), simde_mm_loadl_epi64((simde__m128i *)(tempPtr + 8 * count + 0x160)));
                    horiz_6 = simde_mm_unpacklo_epi16(simde_mm_loadl_epi64((simde__m128i *)(tempPtr + 8 * count + 0x180)), simde_mm_loadl_epi64((simde__m128i *)(tempPtr + 8 * count + 0x1A0)));
                    horiz_7 = simde_mm_unpacklo_epi16(simde_mm_loadl_epi64((simde__m128i *)(tempPtr + 8 * count + 0x1C0)), simde_mm_loadl_epi64((simde__m128i *)(tempPtr + 8 * count + 0x1E0)));
                     
                    //Unpack to re-arrange the packing to the horizontal packing
                    MACRO_UNPACK(32, horiz_0, horiz_1, horiz_2, horiz_3, horiz_4, horiz_5, horiz_6, horiz_7, horiz_01h, horiz_23h, horiz_45h, horiz_67h)
                    MACRO_UNPACK(64, horiz_0, horiz_2, horiz_01h, horiz_23h, horiz_4, horiz_6, horiz_45h, horiz_67h, horiz_0123l, horiz_0123h, horiz_4567l, horiz_4567h)

                    simde_mm_storeu_si128((simde__m128i *)(predPtr), horiz_0);
                    simde_mm_storeu_si128((simde__m128i *)(predPtr + 8), horiz_4);
                    simde_mm_storeu_si128((simde__m128i *)(predPtr + pStride), horiz_0123l);
                    simde_mm_storeu_si128((simde__m128i *)(predPtr + pStride + 8), horiz_4567l);
                    simde_mm_storeu_si128((simde__m128i *)(predPtr + 2 * pStride), horiz_01h);
                    simde_mm_storeu_si128((simde__m128i *)(predPtr + 2 * pStride + 8), horiz_45h);
                    simde_mm_storeu_si128((simde__m128i *)(predPtr + 3 * pStride), horiz_0123h);
                    simde_mm_storeu_si128((simde__m128i *)(predPtr + 3 * pStride + 8), horiz_4567h);
                    predPtr += 4 * pStride;
                }
                tempPtr = temp + 0x200;
                predPtr = predictionPtr + 16;
            }
        }
        else if (size == 16) {

           EB_S8 temp[0x100];
            
            simde__m128i refSamp_0_15, refSamp_1_16;

            for (count = 0; count < 32; count+=2) {
                deltaSum += intraPredAngle;
                deltaInt = (deltaSum >> 5);

                refSamp_0_15 = simde_mm_packs_epi32(simde_mm_and_si128(simde_mm_loadu_si128((simde__m128i *)(refSampMain + deltaInt)), skip_mask), simde_mm_and_si128(simde_mm_loadu_si128((simde__m128i *)(refSampMain + deltaInt + 8)), skip_mask));
                refSamp_1_16 = simde_mm_packs_epi32(simde_mm_and_si128(simde_mm_loadu_si128((simde__m128i *)(refSampMain + deltaInt + 1)), skip_mask), simde_mm_and_si128(simde_mm_loadu_si128((simde__m128i *)(refSampMain + deltaInt + 9)), skip_mask));
                deltaFract = simde_mm_cvtsi32_si128(deltaSum & 31);
                deltaFract = simde_mm_unpacklo_epi16(deltaFract, deltaFract);
                deltaFract = simde_mm_unpacklo_epi32(deltaFract, deltaFract);
                deltaFract = simde_mm_unpacklo_epi64(deltaFract, deltaFract);

                refSampCoeff = simde_mm_sub_epi16(xmm_C32, deltaFract);
                
                vertical_temp = simde_mm_srai_epi16(simde_mm_add_epi16(simde_mm_mullo_epi16(refSamp_0_15, refSampCoeff), simde_mm_add_epi16(simde_mm_mullo_epi16(refSamp_1_16, deltaFract), xmm_C16)), 5);
                
                //The order in the result is not horizontal, store in temp buffer and re-arrange later.
                simde_mm_storeu_si128((simde__m128i *)(temp+8*count), vertical_temp);
            }

            for (count = 0; count < 2; ++count) {

                horiz_0 = simde_mm_unpacklo_epi16(simde_mm_loadl_epi64((simde__m128i *)(temp + 8 * count)), simde_mm_loadl_epi64((simde__m128i *)(temp + 8 * count + 0x10)));
                horiz_1 = simde_mm_unpacklo_epi16(simde_mm_loadl_epi64((simde__m128i *)(temp + 8 * count + 0x20)), simde_mm_loadl_epi64((simde__m128i *)(temp + 8 * count + 0x30)));
                horiz_2 = simde_mm_unpacklo_epi16(simde_mm_loadl_epi64((simde__m128i *)(temp + 8 * count + 0x40)), simde_mm_loadl_epi64((simde__m128i *)(temp + 8 * count + 0x50)));
                horiz_3 = simde_mm_unpacklo_epi16(simde_mm_loadl_epi64((simde__m128i *)(temp + 8 * count + 0x60)), simde_mm_loadl_epi64((simde__m128i *)(temp + 8 * count + 0x70)));
                horiz_4 = simde_mm_unpacklo_epi16(simde_mm_loadl_epi64((simde__m128i *)(temp + 8 * count + 0x80)), simde_mm_loadl_epi64((simde__m128i *)(temp + 8 * count + 0x90)));
                horiz_5 = simde_mm_unpacklo_epi16(simde_mm_loadl_epi64((simde__m128i *)(temp + 8 * count + 0xA0)), simde_mm_loadl_epi64((simde__m128i *)(temp + 8 * count + 0xB0)));
                horiz_6 = simde_mm_unpacklo_epi16(simde_mm_loadl_epi64((simde__m128i *)(temp + 8 * count + 0xC0)), simde_mm_loadl_epi64((simde__m128i *)(temp + 8 * count + 0xD0)));
                horiz_7 = simde_mm_unpacklo_epi16(simde_mm_loadl_epi64((simde__m128i *)(temp + 8 * count + 0xE0)), simde_mm_loadl_epi64((simde__m128i *)(temp + 8 * count + 0xF0)));
                 
                //Unpack to re-arrange the packing to the horizontal packing
                MACRO_UNPACK(32, horiz_0, horiz_1, horiz_2, horiz_3, horiz_4, horiz_5, horiz_6, horiz_7, horiz_01h, horiz_23h, horiz_45h, horiz_67h)
                MACRO_UNPACK(64, horiz_0, horiz_2, horiz_01h, horiz_23h, horiz_4, horiz_6, horiz_45h, horiz_67h, horiz_0123l, horiz_0123h, horiz_4567l, horiz_4567h)

                simde_mm_storeu_si128((simde__m128i *)predictionPtr, horiz_0);
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 8), horiz_4);
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + pStride), horiz_0123l);
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + pStride + 8), horiz_4567l);
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 2 * pStride), horiz_01h);
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 2 * pStride + 8), horiz_45h);
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 3 * pStride), horiz_0123h);
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 3 * pStride + 8), horiz_4567h);
                predictionPtr += 4 * pStride;
            }
        }
        else if (size == 8) {
            
            EB_S8 temp[0x40];
            simde__m128i temp0, temp1, temp2, temp3, temp4, temp5, temp6, temp7, ref0, ref1;
            
            for (count = 0; count < 8; ++count) {
                deltaSum += intraPredAngle;
                deltaInt = (deltaSum >> 5);

                ref0 = simde_mm_and_si128(simde_mm_loadu_si128((simde__m128i *)(refSampMain + deltaInt)), skip_mask);
                ref1 = simde_mm_and_si128(simde_mm_loadu_si128((simde__m128i *)(refSampMain + deltaInt + 1)), skip_mask);
                ref0 = simde_mm_packs_epi32(ref0, ref0);
                ref1 = simde_mm_packs_epi32(ref1, ref1);

                deltaFract = simde_mm_cvtsi32_si128(deltaSum & 31);
                deltaFract = simde_mm_unpacklo_epi16(deltaFract, deltaFract);
                deltaFract = simde_mm_unpacklo_epi32(deltaFract, deltaFract);
                deltaFract = simde_mm_unpacklo_epi64(deltaFract, deltaFract);
                
                refSampCoeff = simde_mm_sub_epi16(xmm_C32, deltaFract);
                
                vertical_temp = simde_mm_srai_epi16(simde_mm_add_epi16(simde_mm_mullo_epi16(ref0, refSampCoeff), simde_mm_add_epi16(simde_mm_mullo_epi16(ref1, deltaFract), xmm_C16)), 5);
                
                //The order in the result is not horizontal, store in temp buffer and re-arrange later.
                simde_mm_storel_epi64((simde__m128i *)(temp+8*count), vertical_temp);
            }
            
            //Unpack to re-arrange the packing to the horizontal packing
            temp0 = simde_mm_unpacklo_epi16(simde_mm_loadl_epi64((simde__m128i *)(temp)), simde_mm_loadl_epi64((simde__m128i *)(temp + 0x08)));
            temp1 = simde_mm_unpacklo_epi16(simde_mm_loadl_epi64((simde__m128i *)(temp + 0x10)), simde_mm_loadl_epi64((simde__m128i *)(temp + 0x18)));
            temp2 = simde_mm_unpacklo_epi16(simde_mm_loadl_epi64((simde__m128i *)(temp + 0x20)), simde_mm_loadl_epi64((simde__m128i *)(temp + 0x28)));
            temp3 = simde_mm_unpacklo_epi16(simde_mm_loadl_epi64((simde__m128i *)(temp + 0x30)), simde_mm_loadl_epi64((simde__m128i *)(temp + 0x38)));
            
            temp4 = simde_mm_unpacklo_epi32(temp0, temp1);
            temp5 = simde_mm_unpackhi_epi32(temp0, temp1);
            temp6 = simde_mm_unpacklo_epi32(temp2, temp3);
            temp7 = simde_mm_unpackhi_epi32(temp2, temp3);            
            
            simde_mm_storeu_si128((simde__m128i *)predictionPtr, simde_mm_unpacklo_epi64(temp4, temp6));
            simde_mm_storeu_si128((simde__m128i *)(predictionPtr + pStride), simde_mm_unpackhi_epi64(temp4, temp6));
            simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 2 * pStride), simde_mm_unpacklo_epi64(temp5, temp7));
            simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 3 * pStride), simde_mm_unpackhi_epi64(temp5, temp7));
        }
        else {
            
            for (count = 0; count < 4; ++count) {
                EB_U64 horiz;
                deltaSum += intraPredAngle;
                deltaInt = (deltaSum >> 5);
                
                deltaFract = simde_mm_cvtsi32_si128(deltaSum & 31);
                deltaFract = simde_mm_unpacklo_epi16(deltaFract, deltaFract);
                deltaFract = simde_mm_unpacklo_epi32(deltaFract, deltaFract);

                refSampCoeff = simde_mm_sub_epi16(xmm_C32, deltaFract);

                vertical_temp = simde_mm_srai_epi16(simde_mm_add_epi16(simde_mm_mullo_epi16(simde_mm_and_si128(simde_mm_loadl_epi64((simde__m128i *)(refSampMain + deltaInt)), skip_mask), refSampCoeff), simde_mm_add_epi16(simde_mm_mullo_epi16(simde_mm_and_si128(simde_mm_loadl_epi64((simde__m128i *)(refSampMain + deltaInt + 1)), skip_mask), deltaFract), xmm_C16)), 5);
                
                horiz = simde_mm_cvtsi128_si64(vertical_temp);
                *predictionPtr = (EB_U16)horiz;
                *(predictionPtr + pStride) = (EB_U16)(horiz >> 32);
                predictionPtr += 1;
            }
        }
    }
}