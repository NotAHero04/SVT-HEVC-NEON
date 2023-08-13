/*
* Copyright(c) 2018 Intel Corporation
* SPDX - License - Identifier: BSD - 2 - Clause - Patent
*/

#include "EbDefinitions.h"
#include "../../../simde/simde/x86/sse4.1.h"

EB_EXTERN void IntraModeDCLuma16bit_SSE4_1_INTRIN(
    const EB_U32   size,                       //input parameter, denotes the size of the current PU
    const EB_U16   *refSamples,                 //input parameter, pointer to the reference samples
    EB_U16         *predictionPtr,              //output parameter, pointer to the prediction
    const EB_U32   predictionBufferStride,     //input parameter, denotes the stride for the prediction ptr
    const EB_BOOL  skip)                       //skip half rows
{
    EB_U32 topOffset = (size << 1) + 1;
    EB_U32 i;
    EB_U32 pStride = skip ? (predictionBufferStride << 1) : predictionBufferStride;

    if (size == 4) {
        simde__m128i sum = simde_mm_setr_epi16(4, 0, 0, 0, 0, 0, 0, 0);
        simde__m128i leftT = simde_mm_loadl_epi64((simde__m128i *)refSamples);
        simde__m128i topL = simde_mm_loadl_epi64((simde__m128i *)(refSamples+topOffset));
        simde__m128i const2;
        simde__m128i temp0, temp1;
        sum = simde_mm_add_epi16(sum, topL);
        sum = simde_mm_add_epi16(sum, leftT);
        sum = simde_mm_hadd_epi16(sum, sum);
        sum = simde_mm_hadd_epi16(sum, sum);
        sum = simde_mm_srli_epi16(sum, 3);
        sum = simde_mm_unpacklo_epi16(sum, sum);
        sum = simde_mm_unpacklo_epi32(sum, sum);

        temp0 = simde_mm_add_epi16(sum, sum);      // 2*predictionPtr[columnIndex]
        temp1 = simde_mm_add_epi16(leftT, topL);   // refSamples[leftOffset] + refSamples[topOffset]
        temp1 = simde_mm_add_epi16(temp1, temp0);  // refSamples[leftOffset] + refSamples[topOffset] + (predictionPtr[0] << 1)
        temp0 = simde_mm_add_epi16(sum, temp0);    // 3*predictionPtr[columnIndex]
        topL = simde_mm_add_epi16(topL, temp0);    // refSamples[topOffset+columnIndex] + 3*predictionPtr[columnIndex]
        leftT = simde_mm_add_epi16(leftT, temp0);  // refSamples[leftOffset+rowIndex] + 3*predictionPtr[writeIndex]
        const2 = simde_mm_setr_epi16(2, 2, 2, 2, 2, 2, 2, 2);
        topL = simde_mm_add_epi16(topL, const2);   // refSamples[topOffset+columnIndex] + 3*predictionPtr[columnIndex] + 2
        leftT = simde_mm_add_epi16(leftT, const2); // refSamples[leftOffset+rowIndex] + 3*predictionPtr[writeIndex] + 2
        temp1 = simde_mm_add_epi16(temp1, const2); // refSamples[leftOffset] + refSamples[topOffset] + (predictionPtr[0] << 1) + 2
        topL = simde_mm_srli_epi16(topL, 2);       // (refSamples[topOffset+columnIndex] + 3*predictionPtr[columnIndex] + 2) >> 2
        leftT = simde_mm_srli_epi16(leftT, 2);     // (refSamples[leftOffset+rowIndex] + 3*predictionPtr[writeIndex] + 2) >> 2
        temp1 = simde_mm_srli_epi16(temp1, 2);     // (refSamples[leftOffset] + refSamples[topOffset] + (predictionPtr[0] << 1) + 2) >> 2

        topL = simde_mm_blend_epi16(topL, temp1, 1); // predictionPtr[0] = (EB_U16) ((refSamples[leftOffset] + refSamples[topOffset] + (predictionPtr[0] << 1) + 2) >> 2);
        simde_mm_storel_epi64((simde__m128i *)predictionPtr, topL); // predictionPtr[columnIndex] = (EB_U16) ((refSamples[topOffset+columnIndex] + 3*predictionPtr[columnIndex] + 2) >> 2);

        if (skip) {
            leftT = simde_mm_srli_si128(leftT, 4);
            topL = simde_mm_blend_epi16(sum, leftT, 1);
            simde_mm_storel_epi64((simde__m128i *)(predictionPtr+pStride), topL);
        }
        else {
            leftT = simde_mm_srli_si128(leftT, 2);
            topL = simde_mm_blend_epi16(sum, leftT, 1);
            simde_mm_storel_epi64((simde__m128i *)(predictionPtr+pStride), topL);

            leftT = simde_mm_srli_si128(leftT, 2);
            topL = simde_mm_blend_epi16(sum, leftT, 1);
            predictionPtr += 2*pStride;
            simde_mm_storel_epi64((simde__m128i *)predictionPtr, topL);
            leftT = simde_mm_srli_si128(leftT, 2);
            topL = simde_mm_blend_epi16(sum, leftT, 1);
            simde_mm_storel_epi64((simde__m128i *)(predictionPtr+pStride), topL);
        }
    }
    else if (size == 8) {
        simde__m128i sum = simde_mm_setr_epi16(8, 0, 0, 0, 0, 0, 0, 0);
        simde__m128i leftT = simde_mm_loadu_si128((simde__m128i *)refSamples);
        simde__m128i topL = simde_mm_loadu_si128((simde__m128i *)(refSamples+topOffset));
        simde__m128i const2;
        simde__m128i temp0, temp1;
        sum = simde_mm_add_epi16(sum, topL);
        sum = simde_mm_add_epi16(sum, leftT);
        sum = simde_mm_hadd_epi16(sum, sum);
        sum = simde_mm_hadd_epi16(sum, sum);
        sum = simde_mm_hadd_epi16(sum, sum);
        sum = simde_mm_srli_epi16(sum, 4);
        sum = simde_mm_unpacklo_epi16(sum, sum);
        sum = simde_mm_unpacklo_epi32(sum, sum);
        sum = simde_mm_unpacklo_epi64(sum, sum);

        temp0 = simde_mm_add_epi16(sum, sum);      // 2*predictionPtr[columnIndex]
        temp1 = simde_mm_add_epi16(leftT, topL);   // refSamples[leftOffset] + refSamples[topOffset]
        temp1 = simde_mm_add_epi16(temp1, temp0);  // refSamples[leftOffset] + refSamples[topOffset] + (predictionPtr[0] << 1)
        temp0 = simde_mm_add_epi16(sum, temp0);    // 3*predictionPtr[columnIndex]
        topL = simde_mm_add_epi16(topL, temp0);    // refSamples[topOffset+columnIndex] + 3*predictionPtr[columnIndex]
        leftT = simde_mm_add_epi16(leftT, temp0);  // refSamples[leftOffset+rowIndex] + 3*predictionPtr[writeIndex]
        const2 = simde_mm_setr_epi16(2, 2, 2, 2, 2, 2, 2, 2);
        topL = simde_mm_add_epi16(topL, const2);   // refSamples[topOffset+columnIndex] + 3*predictionPtr[columnIndex] + 2
        leftT = simde_mm_add_epi16(leftT, const2); // refSamples[leftOffset+rowIndex] + 3*predictionPtr[writeIndex] + 2
        temp1 = simde_mm_add_epi16(temp1, const2); // refSamples[leftOffset] + refSamples[topOffset] + (predictionPtr[0] << 1) + 2
        topL = simde_mm_srli_epi16(topL, 2);       // (refSamples[topOffset+columnIndex] + 3*predictionPtr[columnIndex] + 2) >> 2
        leftT = simde_mm_srli_epi16(leftT, 2);     // (refSamples[leftOffset+rowIndex] + 3*predictionPtr[writeIndex] + 2) >> 2
        temp1 = simde_mm_srli_epi16(temp1, 2);     // (refSamples[leftOffset] + refSamples[topOffset] + (predictionPtr[0] << 1) + 2) >> 2

        topL = simde_mm_blend_epi16(topL, temp1, 1); // predictionPtr[0] = (EB_U16) ((refSamples[leftOffset] + refSamples[topOffset] + (predictionPtr[0] << 1) + 2) >> 2);
        simde_mm_storeu_si128((simde__m128i *)predictionPtr, topL); // predictionPtr[columnIndex] = (EB_U16) ((refSamples[topOffset+columnIndex] + 3*predictionPtr[columnIndex] + 2) >> 2);

        if (skip) {
            leftT = simde_mm_srli_si128(leftT, 4);
            topL = simde_mm_blend_epi16(sum, leftT, 1);
            simde_mm_storeu_si128((simde__m128i *)(predictionPtr+pStride), topL);

            leftT = simde_mm_srli_si128(leftT, 4);
            topL = simde_mm_blend_epi16(sum, leftT, 1);
            predictionPtr += 2*pStride;
            simde_mm_storeu_si128((simde__m128i *)predictionPtr, topL);
            leftT = simde_mm_srli_si128(leftT, 4);
            topL = simde_mm_blend_epi16(sum, leftT, 1);
            simde_mm_storeu_si128((simde__m128i *)(predictionPtr+pStride), topL);
        }
        else {
            leftT = simde_mm_srli_si128(leftT, 2);
            topL = simde_mm_blend_epi16(sum, leftT, 1);
            simde_mm_storeu_si128((simde__m128i *)(predictionPtr+pStride), topL);

            leftT = simde_mm_srli_si128(leftT, 2);
            topL = simde_mm_blend_epi16(sum, leftT, 1);
            predictionPtr += 2*pStride;
            simde_mm_storeu_si128((simde__m128i *)predictionPtr, topL);
            leftT = simde_mm_srli_si128(leftT, 2);
            topL = simde_mm_blend_epi16(sum, leftT, 1);
            simde_mm_storeu_si128((simde__m128i *)(predictionPtr+pStride), topL);

            leftT = simde_mm_srli_si128(leftT, 2);
            topL = simde_mm_blend_epi16(sum, leftT, 1);
            predictionPtr += 2*pStride;
            simde_mm_storeu_si128((simde__m128i *)predictionPtr, topL);
            leftT = simde_mm_srli_si128(leftT, 2);
            topL = simde_mm_blend_epi16(sum, leftT, 1);
            simde_mm_storeu_si128((simde__m128i *)(predictionPtr+pStride), topL);

            leftT = simde_mm_srli_si128(leftT, 2);
            topL = simde_mm_blend_epi16(sum, leftT, 1);
            predictionPtr += 2*pStride;
            simde_mm_storeu_si128((simde__m128i *)predictionPtr, topL);
            leftT = simde_mm_srli_si128(leftT, 2);
            topL = simde_mm_blend_epi16(sum, leftT, 1);
            simde_mm_storeu_si128((simde__m128i *)(predictionPtr+pStride), topL);
        }
    }
    else if (size == 16) {
        simde__m128i sum = simde_mm_setr_epi16(16, 0, 0, 0, 0, 0, 0, 0);
        simde__m128i leftT = simde_mm_loadu_si128((simde__m128i *)refSamples);
        simde__m128i leftB = simde_mm_loadu_si128((simde__m128i *)(refSamples+8));
        simde__m128i topL = simde_mm_loadu_si128((simde__m128i *)(refSamples+topOffset));
        simde__m128i topR = simde_mm_loadu_si128((simde__m128i *)(refSamples+topOffset+8));
        simde__m128i const2;
        simde__m128i temp0, temp1;
        sum = simde_mm_add_epi16(sum, topL);
        sum = simde_mm_add_epi16(sum, topR);
        sum = simde_mm_add_epi16(sum, leftT);
        sum = simde_mm_add_epi16(sum, leftB);
        sum = simde_mm_hadd_epi16(sum, sum);
        sum = simde_mm_hadd_epi16(sum, sum);
        sum = simde_mm_hadd_epi16(sum, sum);
        sum = simde_mm_srli_epi16(sum, 5);
        sum = simde_mm_unpacklo_epi16(sum, sum);
        sum = simde_mm_unpacklo_epi32(sum, sum);
        sum = simde_mm_unpacklo_epi64(sum, sum);

        temp0 = simde_mm_add_epi16(sum, sum);      // 2*predictionPtr[columnIndex]
        temp1 = simde_mm_add_epi16(leftT, topL);   // refSamples[leftOffset] + refSamples[topOffset]
        temp1 = simde_mm_add_epi16(temp1, temp0);  // refSamples[leftOffset] + refSamples[topOffset] + (predictionPtr[0] << 1)
        temp0 = simde_mm_add_epi16(sum, temp0);    // 3*predictionPtr[columnIndex]
        topL = simde_mm_add_epi16(topL, temp0);    // refSamples[topOffset+columnIndex] + 3*predictionPtr[columnIndex]
        topR = simde_mm_add_epi16(topR, temp0);    // refSamples[topOffset+columnIndex] + 3*predictionPtr[columnIndex]
        leftT = simde_mm_add_epi16(leftT, temp0);  // refSamples[leftOffset+rowIndex] + 3*predictionPtr[writeIndex]
        leftB = simde_mm_add_epi16(leftB, temp0);  // refSamples[leftOffset+rowIndex] + 3*predictionPtr[writeIndex]
        const2 = simde_mm_setr_epi16(2, 2, 2, 2, 2, 2, 2, 2);
        topL = simde_mm_add_epi16(topL, const2);   // refSamples[topOffset+columnIndex] + 3*predictionPtr[columnIndex] + 2
        topR = simde_mm_add_epi16(topR, const2);   // refSamples[topOffset+columnIndex] + 3*predictionPtr[columnIndex] + 2
        leftT = simde_mm_add_epi16(leftT, const2); // refSamples[leftOffset+rowIndex] + 3*predictionPtr[writeIndex] + 2
        leftB = simde_mm_add_epi16(leftB, const2); // refSamples[leftOffset+rowIndex] + 3*predictionPtr[writeIndex] + 2
        temp1 = simde_mm_add_epi16(temp1, const2); // refSamples[leftOffset] + refSamples[topOffset] + (predictionPtr[0] << 1) + 2
        topL = simde_mm_srli_epi16(topL, 2);       // (refSamples[topOffset+columnIndex] + 3*predictionPtr[columnIndex] + 2) >> 2
        topR = simde_mm_srli_epi16(topR, 2);       // (refSamples[topOffset+columnIndex] + 3*predictionPtr[columnIndex] + 2) >> 2
        leftT = simde_mm_srli_epi16(leftT, 2);     // (refSamples[leftOffset+rowIndex] + 3*predictionPtr[writeIndex] + 2) >> 2
        leftB = simde_mm_srli_epi16(leftB, 2);     // (refSamples[leftOffset+rowIndex] + 3*predictionPtr[writeIndex] + 2) >> 2
        temp1 = simde_mm_srli_epi16(temp1, 2);     // (refSamples[leftOffset] + refSamples[topOffset] + (predictionPtr[0] << 1) + 2) >> 2

        topL = simde_mm_blend_epi16(topL, temp1, 1); // predictionPtr[0] = (EB_U16) ((refSamples[leftOffset] + refSamples[topOffset] + (predictionPtr[0] << 1) + 2) >> 2);
        simde_mm_storeu_si128((simde__m128i *)predictionPtr, topL); // predictionPtr[columnIndex] = (EB_U16) ((refSamples[topOffset+columnIndex] + 3*predictionPtr[columnIndex] + 2) >> 2);
        simde_mm_storeu_si128((simde__m128i *)(predictionPtr+8), topR);

        if (skip) {
            leftT = simde_mm_srli_si128(leftT, 4);
            topL = simde_mm_blend_epi16(sum, leftT, 1);
            simde_mm_storeu_si128((simde__m128i *)(predictionPtr+pStride), topL);
            simde_mm_storeu_si128((simde__m128i *)(predictionPtr+pStride+8), sum);

            leftT = simde_mm_srli_si128(leftT, 4);
            topL = simde_mm_blend_epi16(sum, leftT, 1);
            predictionPtr += 2*pStride;
            simde_mm_storeu_si128((simde__m128i *)predictionPtr, topL);
            simde_mm_storeu_si128((simde__m128i *)(predictionPtr+8), sum);
            leftT = simde_mm_srli_si128(leftT, 4);
            topL = simde_mm_blend_epi16(sum, leftT, 1);
            simde_mm_storeu_si128((simde__m128i *)(predictionPtr+pStride), topL);
            simde_mm_storeu_si128((simde__m128i *)(predictionPtr+pStride+8), sum);

            topL = simde_mm_blend_epi16(sum, leftB, 1);
            predictionPtr += 2*pStride;
            simde_mm_storeu_si128((simde__m128i *)predictionPtr, topL);
            simde_mm_storeu_si128((simde__m128i *)(predictionPtr+8), sum);
            leftB = simde_mm_srli_si128(leftB, 4);
            topL = simde_mm_blend_epi16(sum, leftB, 1);
            simde_mm_storeu_si128((simde__m128i *)(predictionPtr+pStride), topL);
            simde_mm_storeu_si128((simde__m128i *)(predictionPtr+pStride+8), sum);

            leftB = simde_mm_srli_si128(leftB, 4);
            topL = simde_mm_blend_epi16(sum, leftB, 1);
            predictionPtr += 2*pStride;
            simde_mm_storeu_si128((simde__m128i *)predictionPtr, topL);
            simde_mm_storeu_si128((simde__m128i *)(predictionPtr+8), sum);
            leftB = simde_mm_srli_si128(leftB, 4);
            topL = simde_mm_blend_epi16(sum, leftB, 1);
            simde_mm_storeu_si128((simde__m128i *)(predictionPtr+pStride), topL);
            simde_mm_storeu_si128((simde__m128i *)(predictionPtr+pStride+8), sum);
        }
        else {
            leftT = simde_mm_srli_si128(leftT, 2);
            topL = simde_mm_blend_epi16(sum, leftT, 1);
            simde_mm_storeu_si128((simde__m128i *)(predictionPtr+pStride), topL);
            simde_mm_storeu_si128((simde__m128i *)(predictionPtr+pStride+8), sum);

            leftT = simde_mm_srli_si128(leftT, 2);
            topL = simde_mm_blend_epi16(sum, leftT, 1);
            predictionPtr += 2*pStride;
            simde_mm_storeu_si128((simde__m128i *)predictionPtr, topL);
            simde_mm_storeu_si128((simde__m128i *)(predictionPtr+8), sum);
            leftT = simde_mm_srli_si128(leftT, 2);
            topL = simde_mm_blend_epi16(sum, leftT, 1);
            simde_mm_storeu_si128((simde__m128i *)(predictionPtr+pStride), topL);
            simde_mm_storeu_si128((simde__m128i *)(predictionPtr+pStride+8), sum);

            leftT = simde_mm_srli_si128(leftT, 2);
            topL = simde_mm_blend_epi16(sum, leftT, 1);
            predictionPtr += 2*pStride;
            simde_mm_storeu_si128((simde__m128i *)predictionPtr, topL);
            simde_mm_storeu_si128((simde__m128i *)(predictionPtr+8), sum);
            leftT = simde_mm_srli_si128(leftT, 2);
            topL = simde_mm_blend_epi16(sum, leftT, 1);
            simde_mm_storeu_si128((simde__m128i *)(predictionPtr+pStride), topL);
            simde_mm_storeu_si128((simde__m128i *)(predictionPtr+pStride+8), sum);

            leftT = simde_mm_srli_si128(leftT, 2);
            topL = simde_mm_blend_epi16(sum, leftT, 1);
            predictionPtr += 2*pStride;
            simde_mm_storeu_si128((simde__m128i *)predictionPtr, topL);
            simde_mm_storeu_si128((simde__m128i *)(predictionPtr+8), sum);
            leftT = simde_mm_srli_si128(leftT, 2);
            topL = simde_mm_blend_epi16(sum, leftT, 1);
            simde_mm_storeu_si128((simde__m128i *)(predictionPtr+pStride), topL);
            simde_mm_storeu_si128((simde__m128i *)(predictionPtr+pStride+8), sum);

            topL = simde_mm_blend_epi16(sum, leftB, 1);
            predictionPtr += 2*pStride;
            simde_mm_storeu_si128((simde__m128i *)predictionPtr, topL);
            simde_mm_storeu_si128((simde__m128i *)(predictionPtr+8), sum);
            leftB = simde_mm_srli_si128(leftB, 2);
            topL = simde_mm_blend_epi16(sum, leftB, 1);
            simde_mm_storeu_si128((simde__m128i *)(predictionPtr+pStride), topL);
            simde_mm_storeu_si128((simde__m128i *)(predictionPtr+pStride+8), sum);

            leftB = simde_mm_srli_si128(leftB, 2);
            topL = simde_mm_blend_epi16(sum, leftB, 1);
            predictionPtr += 2*pStride;
            simde_mm_storeu_si128((simde__m128i *)predictionPtr, topL);
            simde_mm_storeu_si128((simde__m128i *)(predictionPtr+8), sum);
            leftB = simde_mm_srli_si128(leftB, 2);
            topL = simde_mm_blend_epi16(sum, leftB, 1);
            simde_mm_storeu_si128((simde__m128i *)(predictionPtr+pStride), topL);
            simde_mm_storeu_si128((simde__m128i *)(predictionPtr+pStride+8), sum);

            leftB = simde_mm_srli_si128(leftB, 2);
            topL = simde_mm_blend_epi16(sum, leftB, 1);
            predictionPtr += 2*pStride;
            simde_mm_storeu_si128((simde__m128i *)predictionPtr, topL);
            simde_mm_storeu_si128((simde__m128i *)(predictionPtr+8), sum);
            leftB = simde_mm_srli_si128(leftB, 2);
            topL = simde_mm_blend_epi16(sum, leftB, 1);
            simde_mm_storeu_si128((simde__m128i *)(predictionPtr+pStride), topL);
            simde_mm_storeu_si128((simde__m128i *)(predictionPtr+pStride+8), sum);

            leftB = simde_mm_srli_si128(leftB, 2);
            topL = simde_mm_blend_epi16(sum, leftB, 1);
            predictionPtr += 2*pStride;
            simde_mm_storeu_si128((simde__m128i *)predictionPtr, topL);
            simde_mm_storeu_si128((simde__m128i *)(predictionPtr+8), sum);
            leftB = simde_mm_srli_si128(leftB, 2);
            topL = simde_mm_blend_epi16(sum, leftB, 1);
            simde_mm_storeu_si128((simde__m128i *)(predictionPtr+pStride), topL);
            simde_mm_storeu_si128((simde__m128i *)(predictionPtr+pStride+8), sum);
        }
    }
    else { /*if (size == 32) {*/
        simde__m128i sum = simde_mm_setr_epi16(32, 0, 0, 0, 0, 0, 0, 0);
        sum = simde_mm_add_epi16(sum, simde_mm_loadu_si128((simde__m128i *)refSamples));
        sum = simde_mm_add_epi16(sum, simde_mm_loadu_si128((simde__m128i *)(refSamples+8)));
        sum = simde_mm_add_epi16(sum, simde_mm_loadu_si128((simde__m128i *)(refSamples+16)));
        sum = simde_mm_add_epi16(sum, simde_mm_loadu_si128((simde__m128i *)(refSamples+24)));
        sum = simde_mm_add_epi16(sum, simde_mm_loadu_si128((simde__m128i *)(refSamples+topOffset)));
        sum = simde_mm_add_epi16(sum, simde_mm_loadu_si128((simde__m128i *)(refSamples+topOffset+8)));
        sum = simde_mm_add_epi16(sum, simde_mm_loadu_si128((simde__m128i *)(refSamples+topOffset+16)));
        sum = simde_mm_add_epi16(sum, simde_mm_loadu_si128((simde__m128i *)(refSamples+topOffset+24)));
        sum = simde_mm_hadd_epi16(sum, sum);
        sum = simde_mm_hadd_epi16(sum, sum);
        sum = simde_mm_hadd_epi16(sum, sum);
        sum = simde_mm_srli_epi16(sum, 6);
        sum = simde_mm_unpacklo_epi16(sum, sum);
        sum = simde_mm_unpacklo_epi32(sum, sum);
        sum = simde_mm_unpacklo_epi64(sum, sum);
        for (i=0; i<(EB_U32)(skip ? 8: 16); i++) {
            simde_mm_storeu_si128((simde__m128i *)predictionPtr, sum);
            simde_mm_storeu_si128((simde__m128i *)(predictionPtr+8), sum);
            simde_mm_storeu_si128((simde__m128i *)(predictionPtr+16), sum);
            simde_mm_storeu_si128((simde__m128i *)(predictionPtr+24), sum);
            simde_mm_storeu_si128((simde__m128i *)(predictionPtr+pStride), sum);
            simde_mm_storeu_si128((simde__m128i *)(predictionPtr+pStride+8), sum);
            simde_mm_storeu_si128((simde__m128i *)(predictionPtr+pStride+16), sum);
            simde_mm_storeu_si128((simde__m128i *)(predictionPtr+pStride+24), sum);
            predictionPtr += 2 * pStride;
        }
    }
}
