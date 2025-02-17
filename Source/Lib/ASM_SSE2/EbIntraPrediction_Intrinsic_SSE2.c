/*
* Copyright(c) 2018 Intel Corporation
* SPDX - License - Identifier: BSD - 2 - Clause - Patent
*/

#include "EbDefinitions.h"
#include "../../../simde/simde/x86/sse2.h"
#include "EbMcp_SSE2.h"
#include "EbIntraPrediction_SSE2.h"

#define OFFSET_17TO31   0
#define OFFSET_1TO15    (8+OFFSET_17TO31)
#define OFFSET_25TO32   (8+OFFSET_1TO15)
#define OFFSET_17TO24   (8+OFFSET_25TO32)
#define OFFSET_9TO16    (8+OFFSET_17TO24)
#define OFFSET_1TO8     (8+OFFSET_9TO16)
#define OFFSET_31TO24   (8+OFFSET_1TO8)
#define OFFSET_23TO16   (8+OFFSET_31TO24)
#define OFFSET_15TO8    (8+OFFSET_23TO16)
#define OFFSET_7TO0     (8+OFFSET_15TO8)
#define OFFSET_3TO0     (4+OFFSET_7TO0) // not a separate entry
#define OFFSET_0        (8+OFFSET_7TO0)
#define OFFSET_1        (8+OFFSET_0)
#define OFFSET_2        (8+OFFSET_1)
#define OFFSET_3        (8+OFFSET_2)
#define OFFSET_4        (8+OFFSET_3)
#define OFFSET_5        (8+OFFSET_4)
#define OFFSET_6        (8+OFFSET_5)
#define OFFSET_7        (8+OFFSET_6)
#define OFFSET_8        (8+OFFSET_7)
#define OFFSET_9        (8+OFFSET_8)
#define OFFSET_10       (8+OFFSET_9)
#define OFFSET_11       (8+OFFSET_10)
#define OFFSET_12       (8+OFFSET_11)
#define OFFSET_13       (8+OFFSET_12)
#define OFFSET_14       (8+OFFSET_13)
#define OFFSET_15       (8+OFFSET_14)
#define OFFSET_16       (8+OFFSET_15)
#define OFFSET_17       (8+OFFSET_16)
#define OFFSET_18       (8+OFFSET_17)
#define OFFSET_19       (8+OFFSET_18)
#define OFFSET_20       (8+OFFSET_19)
#define OFFSET_21       (8+OFFSET_20)
#define OFFSET_22       (8+OFFSET_21)
#define OFFSET_23       (8+OFFSET_22)
#define OFFSET_24       (8+OFFSET_23)
#define OFFSET_25       (8+OFFSET_24)
#define OFFSET_26       (8+OFFSET_25)
#define OFFSET_27       (8+OFFSET_26)
#define OFFSET_28       (8+OFFSET_27)
#define OFFSET_29       (8+OFFSET_28)
#define OFFSET_30       (8+OFFSET_29)
#define OFFSET_31       (8+OFFSET_30)
#define OFFSET_32       (8+OFFSET_31)

#define MACRO_VERTICAL_LUMA_4(A, B, C) \
    *(EB_U32*)predictionPtr = simde_mm_cvtsi128_si32(simde_mm_or_si128(simde_mm_and_si128(A, B), C)); \
    A = simde_mm_srli_si128(A, 1); \
    *(EB_U32*)(predictionPtr + pStride) = simde_mm_cvtsi128_si32(simde_mm_or_si128(simde_mm_and_si128(A, B), C)); \
    A = simde_mm_srli_si128(A, 1);

#define MACRO_HORIZONTAL_LUMA_32X16(A)\
    left0_14_even = simde_mm_and_si128(simde_mm_loadu_si128((simde__m128i *)(refSamples+leftOffset+A)), skip_mask);\
    left0_14_even = simde_mm_packus_epi16(left0_14_even, left0_14_even);\
    left0_14_even = simde_mm_unpacklo_epi8(left0_14_even, left0_14_even);\
    left0_6_even = simde_mm_unpacklo_epi16(left0_14_even, left0_14_even);\
    left8_14_even = simde_mm_unpackhi_epi16(left0_14_even, left0_14_even);\
    left02 = simde_mm_unpacklo_epi32(left0_6_even, left0_6_even);\
    left46 = simde_mm_unpackhi_epi32(left0_6_even, left0_6_even);\
    left8_10 = simde_mm_unpacklo_epi32(left8_14_even, left8_14_even);\
    left12_14 = simde_mm_unpackhi_epi32(left8_14_even, left8_14_even);\
    left0 = simde_mm_unpacklo_epi64(left02, left02);\
    left2 = simde_mm_unpackhi_epi64(left02, left02);\
    left4 = simde_mm_unpacklo_epi64(left46, left46);\
    left6 = simde_mm_unpackhi_epi64(left46, left46);\
    left8 = simde_mm_unpacklo_epi64(left8_10, left8_10);\
    left10 = simde_mm_unpackhi_epi64(left8_10, left8_10);\
    left12 = simde_mm_unpacklo_epi64(left12_14, left12_14);\
    left14 = simde_mm_unpackhi_epi64(left12_14, left12_14);\
    simde_mm_storeu_si128((simde__m128i *)(predictionPtr), left0);\
    simde_mm_storeu_si128((simde__m128i *)(predictionPtr+16), left0);\
    simde_mm_storeu_si128((simde__m128i *)(predictionPtr+pStride), left2);\
    simde_mm_storeu_si128((simde__m128i *)(predictionPtr+pStride+16), left2);\
    simde_mm_storeu_si128((simde__m128i *)(predictionPtr+2*pStride), left4);\
    simde_mm_storeu_si128((simde__m128i *)(predictionPtr+2*pStride+16), left4);\
    simde_mm_storeu_si128((simde__m128i *)(predictionPtr+3*pStride), left6);\
    simde_mm_storeu_si128((simde__m128i *)(predictionPtr+3*pStride+16), left6);\
    predictionPtr += (pStride << 2);\
    simde_mm_storeu_si128((simde__m128i *)(predictionPtr), left8);\
    simde_mm_storeu_si128((simde__m128i *)(predictionPtr+16), left8);\
    simde_mm_storeu_si128((simde__m128i *)(predictionPtr+pStride), left10);\
    simde_mm_storeu_si128((simde__m128i *)(predictionPtr+pStride+16), left10);\
    simde_mm_storeu_si128((simde__m128i *)(predictionPtr+2*pStride), left12);\
    simde_mm_storeu_si128((simde__m128i *)(predictionPtr+2*pStride+16), left12);\
    simde_mm_storeu_si128((simde__m128i *)(predictionPtr+3*pStride), left14);\
    simde_mm_storeu_si128((simde__m128i *)(predictionPtr+3*pStride+16), left14);


void IntraModeHorizontalLuma_SSE2_INTRIN(
    const EB_U32      size,                   //input parameter, denotes the size of the current PU
    EB_U8            *refSamples,             //input parameter, pointer to the reference samples
    EB_U8            *predictionPtr,          //output parameter, pointer to the prediction
    const EB_U32      predictionBufferStride, //input parameter, denotes the stride for the prediction ptr
    const EB_BOOL     skip)                    //skip one row 
{
    EB_U32 topOffset = (size << 1) + 1;
    EB_U32 topLeftOffset = (size << 1);
    EB_U32 leftOffset = 0;
    EB_U32 pStride = predictionBufferStride;

    simde__m128i xmm_0, top, topLeft, filter_cmpnt0_7, unclipped, clipped; 
    simde__m128i left0_15, left0_7, left8_15, left0_3, left4_7, left8_11, left12_15, left01, left23, left45, left67, left89, left10_11;
    simde__m128i left12_13, left14_15, left0, left1, left2, left3, left4, left5, left6, left7, left8, left9, left10, left11;
    simde__m128i left12, left13, left14, left15;
    
    if (size != 32) {    

        topLeft = simde_mm_cvtsi32_si128(*(refSamples + topLeftOffset));
        xmm_0 = simde_mm_setzero_si128();
        topLeft = simde_mm_unpacklo_epi8(topLeft, xmm_0);
        topLeft = simde_mm_unpacklo_epi16(topLeft, topLeft);
        topLeft = simde_mm_unpacklo_epi32(topLeft, topLeft);
        topLeft = simde_mm_unpacklo_epi64(topLeft, topLeft);

        if (!skip) {

            if (size == 16) {
                simde__m128i filter_cmpnt8_15;
                
                top = simde_mm_loadu_si128((simde__m128i *)(refSamples + topOffset));
                
                filter_cmpnt0_7 = simde_mm_srai_epi16(simde_mm_sub_epi16(simde_mm_unpacklo_epi8(top, xmm_0), topLeft), 1);
                filter_cmpnt8_15 = simde_mm_srai_epi16(simde_mm_sub_epi16(simde_mm_unpackhi_epi8(top, xmm_0), topLeft), 1);

                left0_15 = simde_mm_loadu_si128((simde__m128i *)(refSamples + leftOffset));
                left0_7 = simde_mm_unpacklo_epi8(left0_15, left0_15);
                left8_15 = simde_mm_unpackhi_epi8(left0_15, left0_15);
                
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
                
                left0 = simde_mm_unpacklo_epi8(left01, xmm_0);                
                clipped = simde_mm_packus_epi16(simde_mm_add_epi16(left0, filter_cmpnt0_7), simde_mm_add_epi16(filter_cmpnt8_15, left0));
                
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

                simde_mm_storeu_si128((simde__m128i *)predictionPtr, clipped);
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + pStride), left1);
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr+2*pStride), left2);
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr+3*pStride), left3);
                predictionPtr += (pStride << 2);
                simde_mm_storeu_si128((simde__m128i *)predictionPtr, left4);
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + pStride), left5);
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr+2*pStride), left6);
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr+3*pStride), left7);
                predictionPtr += (pStride << 2);
                simde_mm_storeu_si128((simde__m128i *)predictionPtr, left8);
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + pStride), left9);
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr+2*pStride), left10);
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr+3*pStride), left11);
                predictionPtr += (pStride << 2);                
                simde_mm_storeu_si128((simde__m128i *)predictionPtr, left12);
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + pStride), left13);
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr+2*pStride), left14);
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr+3*pStride), left15);
            }
            else if (size == 8) {          
                
                filter_cmpnt0_7 = simde_mm_srai_epi16(simde_mm_sub_epi16(simde_mm_unpacklo_epi8(simde_mm_loadl_epi64((simde__m128i *)(refSamples + topOffset)), xmm_0), topLeft), 1);

                left0_7 = simde_mm_loadl_epi64((simde__m128i *)(refSamples + leftOffset));
                left0_7 = simde_mm_unpacklo_epi8(left0_7, left0_7);
                
                left0_3 = simde_mm_unpacklo_epi16(left0_7, left0_7);
                left4_7 = simde_mm_unpackhi_epi16(left0_7, left0_7);
                
                left01 = simde_mm_unpacklo_epi32(left0_3, left0_3);
                left23 = simde_mm_unpackhi_epi32(left0_3, left0_3);
                left45 = simde_mm_unpacklo_epi32(left4_7, left4_7);
                left67 = simde_mm_unpackhi_epi32(left4_7, left4_7);
                
                unclipped = simde_mm_add_epi16(simde_mm_unpacklo_epi8(left01, xmm_0), filter_cmpnt0_7);
                clipped = simde_mm_packus_epi16(unclipped, unclipped);

                simde_mm_storel_epi64((simde__m128i *)predictionPtr, clipped);
                simde_mm_storel_epi64((simde__m128i *)(predictionPtr + pStride), simde_mm_srli_si128(left01, 8));
                simde_mm_storel_epi64((simde__m128i *)(predictionPtr + 2 * pStride), left23);
                simde_mm_storel_epi64((simde__m128i *)(predictionPtr + 3 * pStride), simde_mm_srli_si128(left23, 8));
                predictionPtr += (pStride << 2);
                simde_mm_storel_epi64((simde__m128i *)predictionPtr, left45);
                simde_mm_storel_epi64((simde__m128i *)(predictionPtr + pStride), simde_mm_srli_si128(left45, 8));
                simde_mm_storel_epi64((simde__m128i *)(predictionPtr + 2 * pStride), left67);
                simde_mm_storel_epi64((simde__m128i *)(predictionPtr + 3 * pStride), simde_mm_srli_si128(left67, 8));

            } else {           
                simde__m128i filter_cmpnt0_3;

                filter_cmpnt0_3 = simde_mm_srai_epi16(simde_mm_sub_epi16(simde_mm_unpacklo_epi8(simde_mm_cvtsi32_si128(*(EB_U32*)(refSamples + topOffset)), xmm_0), topLeft), 1);          
                 
                left0_3 = simde_mm_cvtsi32_si128(*(EB_U32*)(refSamples + leftOffset)); 
                left0_3 = simde_mm_unpacklo_epi8(left0_3, left0_3);  //00112233
                left0_3 = simde_mm_unpacklo_epi16(left0_3, left0_3); //0000111122223333
                
                left01 = simde_mm_unpacklo_epi32(left0_3, left0_3); 
                left23 = simde_mm_unpackhi_epi32(left0_3, left0_3); 

                unclipped = simde_mm_add_epi16(simde_mm_unpacklo_epi8(left01, xmm_0), filter_cmpnt0_3);      
                clipped = simde_mm_packus_epi16(unclipped, unclipped);   
                
                *(EB_U32*)predictionPtr = simde_mm_cvtsi128_si32(clipped);
                *(EB_U32*)(predictionPtr + pStride) = simde_mm_cvtsi128_si32(simde_mm_srli_si128(left01, 8));     
                *(EB_U32*)(predictionPtr + 2 * pStride) = simde_mm_cvtsi128_si32(left23); 
                *(EB_U32*)(predictionPtr + 3 * pStride) = simde_mm_cvtsi128_si32(simde_mm_srli_si128(left23, 8));    
            }
        }
        else {
            simde__m128i left0_14_even, left_0_6_even, left02, left46, skip_mask;
            
            skip_mask = simde_mm_set1_epi16(0x00FF); 
            pStride <<= 1; 
            
            if (size == 16) {
                simde__m128i filter_cmpnt8_15, left_8_14_even, left8_10, left12_14, left02_16wide;

                top = simde_mm_loadu_si128((simde__m128i *)(refSamples + topOffset));
                filter_cmpnt0_7 = simde_mm_srai_epi16(simde_mm_sub_epi16(simde_mm_unpacklo_epi8(top, xmm_0), topLeft), 1);                
                filter_cmpnt8_15 = simde_mm_srai_epi16(simde_mm_sub_epi16(simde_mm_unpackhi_epi8(top, xmm_0), topLeft), 1);
                
                left0_14_even = simde_mm_and_si128(simde_mm_loadu_si128((simde__m128i *)(refSamples + leftOffset)), skip_mask);
                left0_14_even = simde_mm_packus_epi16(left0_14_even, left0_14_even);
                left0_14_even = simde_mm_unpacklo_epi8(left0_14_even, left0_14_even);

                left_0_6_even = simde_mm_unpacklo_epi16(left0_14_even, left0_14_even);
                left_8_14_even = simde_mm_unpackhi_epi16(left0_14_even, left0_14_even);
                
                left02 = simde_mm_unpacklo_epi32(left_0_6_even, left_0_6_even);
                left46 = simde_mm_unpackhi_epi32(left_0_6_even, left_0_6_even);
                left8_10 = simde_mm_unpacklo_epi32(left_8_14_even, left_8_14_even);
                left12_14 = simde_mm_unpackhi_epi32(left_8_14_even, left_8_14_even);
                
                left02_16wide = simde_mm_unpacklo_epi8(left02, xmm_0);
                clipped = simde_mm_packus_epi16(simde_mm_add_epi16(left02_16wide, filter_cmpnt0_7), simde_mm_add_epi16(filter_cmpnt8_15, left02_16wide));
                
                left2 = simde_mm_unpackhi_epi64(left02, left02);
                left4 = simde_mm_unpacklo_epi64(left46, left46);
                left6 = simde_mm_unpackhi_epi64(left46, left46);
                left8 = simde_mm_unpacklo_epi64(left8_10, left8_10);
                left10 = simde_mm_unpackhi_epi64(left8_10, left8_10);
                left12 = simde_mm_unpacklo_epi64(left12_14, left12_14);
                left14 = simde_mm_unpackhi_epi64(left12_14, left12_14);

                simde_mm_storeu_si128((simde__m128i *)(predictionPtr), clipped);
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + pStride), left2);
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 2 * pStride), left4);
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 3 * pStride), left6);
                predictionPtr += (pStride << 2);
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr), left8);
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + pStride), left10);
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 2 * pStride), left12);
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 3 * pStride), left14);
 
            }
            else {
                filter_cmpnt0_7 = simde_mm_srai_epi16(simde_mm_sub_epi16(simde_mm_unpacklo_epi8(simde_mm_loadl_epi64((simde__m128i *)(refSamples + topOffset)), xmm_0), topLeft), 1);

                left_0_6_even = simde_mm_and_si128(simde_mm_loadl_epi64((simde__m128i *)(refSamples + leftOffset)), skip_mask);
                left_0_6_even = simde_mm_packus_epi16(left_0_6_even, left_0_6_even); 
                left_0_6_even = simde_mm_unpacklo_epi8(left_0_6_even, left_0_6_even);
                left_0_6_even = simde_mm_unpacklo_epi16(left_0_6_even, left_0_6_even);
                
                left02 = simde_mm_unpacklo_epi32(left_0_6_even, left_0_6_even);
                left46 = simde_mm_unpackhi_epi32(left_0_6_even, left_0_6_even);

                unclipped = simde_mm_add_epi16(simde_mm_unpacklo_epi8(left02, xmm_0), filter_cmpnt0_7);
                clipped = simde_mm_packus_epi16(unclipped, unclipped);

                simde_mm_storel_epi64((simde__m128i *)predictionPtr, clipped);
                simde_mm_storel_epi64((simde__m128i *)(predictionPtr + pStride), simde_mm_srli_si128(left02, 8));
                simde_mm_storel_epi64((simde__m128i *)(predictionPtr + 2 * pStride), left46);
                simde_mm_storel_epi64((simde__m128i *)(predictionPtr + 3 * pStride), simde_mm_srli_si128(left46, 8));
            }
        }
    }
    else {
        if (!skip) {

            EB_U8 count;

            for (count = 0; count < 2; ++count) {
                left0_15 = simde_mm_loadu_si128((simde__m128i *)(refSamples + leftOffset));
                refSamples += 16;
                
                left0_7 = simde_mm_unpacklo_epi8(left0_15, left0_15);
                left8_15 = simde_mm_unpackhi_epi8(left0_15, left0_15);
                
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

                simde_mm_storeu_si128((simde__m128i *)predictionPtr, left0);
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 16), left0);
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr+pStride), left1);
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr+pStride+16), left1);
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr+2*pStride), left2);
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr+2*pStride+16), left2);
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr+3*pStride), left3);
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr+3*pStride+16), left3);
                predictionPtr += (pStride << 2);
                 simde_mm_storeu_si128((simde__m128i *)predictionPtr, left4);
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 16), left4);
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr+pStride), left5);
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr+pStride+16), left5);
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr+2*pStride), left6);
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr+2*pStride+16), left6);
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr+3*pStride), left7);
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr+3*pStride+16), left7);
                predictionPtr += (pStride << 2);
                simde_mm_storeu_si128((simde__m128i *)predictionPtr, left8);
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 16), left8);
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr+pStride), left9);
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr+pStride+16), left9);
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr+2*pStride), left10);
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr+2*pStride+16), left10);
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr+3*pStride), left11);
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr+3*pStride+16), left11);
                predictionPtr += (pStride << 2);
                simde_mm_storeu_si128((simde__m128i *)predictionPtr, left12);
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 16), left12);
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr+pStride), left13);
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr+pStride+16), left13);
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr+2*pStride), left14);
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr+2*pStride+16), left14);
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr+3*pStride), left15);
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr+3*pStride+16), left15);
                predictionPtr += (pStride << 2);
            }
        }
        else {
            simde__m128i left0_14_even, left0_6_even, left8_14_even, left8_10, left12_14, left02, left46, skip_mask;
            skip_mask = simde_mm_set1_epi16(0x00FF);
            pStride <<= 1;
            MACRO_HORIZONTAL_LUMA_32X16(0)
            predictionPtr += (pStride << 2);
            MACRO_HORIZONTAL_LUMA_32X16(16)
        }
    }
}


void IntraModeHorizontalChroma_SSE2_INTRIN(
    const EB_U32      size,                   //input parameter, denotes the size of the current PU
    EB_U8            *refSamples,             //input parameter, pointer to the reference samples
    EB_U8            *predictionPtr,          //output parameter, pointer to the prediction
    const EB_U32      predictionBufferStride, //input parameter, denotes the stride for the prediction ptr
    const EB_BOOL     skip)                    //skip one row 
{
    EB_U32 pStride = predictionBufferStride;
    EB_U32 leftOffset = 0;
    //Jing:
    //Add size == 32 for 444
    
    simde__m128i left0_15, left0_7, left8_15, left0_3, left4_7, left8_11, left12_15, left01, left23, left45, left67, left89, left10_11;
    simde__m128i left12_13, left14_15, left0, left1, left2, left3, left4, left5, left6, left7, left8, left9, left10, left11;
    simde__m128i left12, left13, left14, left15;
    if (!skip) {
        if (size == 32) {
            EB_U8 count;
            for (count = 0; count < 2; ++count) {
                left0_15 = simde_mm_loadu_si128((simde__m128i *)(refSamples + leftOffset));
                refSamples += 16;

                left0_7 = simde_mm_unpacklo_epi8(left0_15, left0_15);
                left8_15 = simde_mm_unpackhi_epi8(left0_15, left0_15);

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

                simde_mm_storeu_si128((simde__m128i *)predictionPtr, left0);
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 16), left0);
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr+pStride), left1);
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr+pStride+16), left1);
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr+2*pStride), left2);
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr+2*pStride+16), left2);
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr+3*pStride), left3);
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr+3*pStride+16), left3);
                predictionPtr += (pStride << 2);
                simde_mm_storeu_si128((simde__m128i *)predictionPtr, left4);
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 16), left4);
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr+pStride), left5);
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr+pStride+16), left5);
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr+2*pStride), left6);
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr+2*pStride+16), left6);
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr+3*pStride), left7);
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr+3*pStride+16), left7);
                predictionPtr += (pStride << 2);
                simde_mm_storeu_si128((simde__m128i *)predictionPtr, left8);
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 16), left8);
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr+pStride), left9);
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr+pStride+16), left9);
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr+2*pStride), left10);
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr+2*pStride+16), left10);
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr+3*pStride), left11);
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr+3*pStride+16), left11);
                predictionPtr += (pStride << 2);
                simde_mm_storeu_si128((simde__m128i *)predictionPtr, left12);
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 16), left12);
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr+pStride), left13);
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr+pStride+16), left13);
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr+2*pStride), left14);
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr+2*pStride+16), left14);
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr+3*pStride), left15);
                simde_mm_storeu_si128((simde__m128i *)(predictionPtr+3*pStride+16), left15);
                predictionPtr += (pStride << 2);
            }
        } else if (size == 16) {
            simde__m128i xmm0, xmm2, xmm4, xmm6, xmm8, xmm10, xmm12, xmm14;
            xmm0 = simde_mm_loadu_si128((simde__m128i *)(refSamples+leftOffset));
            xmm8 = simde_mm_unpackhi_epi8(xmm0, xmm0);
            xmm0 = simde_mm_unpacklo_epi8(xmm0, xmm0);
            
            xmm4 = simde_mm_unpackhi_epi16(xmm0, xmm0);
            xmm0 = simde_mm_unpacklo_epi16(xmm0, xmm0);
            xmm12 = simde_mm_unpackhi_epi16(xmm8, xmm8);
            xmm8 = simde_mm_unpacklo_epi16(xmm8, xmm8);
            
            xmm2 = simde_mm_unpackhi_epi32(xmm0, xmm0);
            xmm0 = simde_mm_unpacklo_epi32(xmm0, xmm0);
            xmm6 = simde_mm_unpackhi_epi32(xmm4, xmm4);
            xmm4 = simde_mm_unpacklo_epi32(xmm4, xmm4);
            xmm10 = simde_mm_unpackhi_epi32(xmm8, xmm8);
            xmm8 = simde_mm_unpacklo_epi32(xmm8, xmm8);
            xmm14 = simde_mm_unpackhi_epi32(xmm12, xmm12);
            xmm12 = simde_mm_unpacklo_epi32(xmm12, xmm12);
               
            simde_mm_storeu_si128((simde__m128i *)predictionPtr,             simde_mm_unpacklo_epi64(xmm0, xmm0));
            simde_mm_storeu_si128((simde__m128i *)(predictionPtr + pStride), simde_mm_unpackhi_epi64(xmm0, xmm0));
            simde_mm_storeu_si128((simde__m128i *)(predictionPtr+2*pStride), simde_mm_unpacklo_epi64(xmm2, xmm2));
            simde_mm_storeu_si128((simde__m128i *)(predictionPtr+3*pStride), simde_mm_unpackhi_epi64(xmm2, xmm2));
            predictionPtr += (pStride << 2);
            simde_mm_storeu_si128((simde__m128i *)predictionPtr,             simde_mm_unpacklo_epi64(xmm4, xmm4));
            simde_mm_storeu_si128((simde__m128i *)(predictionPtr + pStride), simde_mm_unpackhi_epi64(xmm4, xmm4));
            simde_mm_storeu_si128((simde__m128i *)(predictionPtr+2*pStride), simde_mm_unpacklo_epi64(xmm6, xmm6));
            simde_mm_storeu_si128((simde__m128i *)(predictionPtr+3*pStride), simde_mm_unpackhi_epi64(xmm6, xmm6));
            predictionPtr += (pStride << 2);
            simde_mm_storeu_si128((simde__m128i *)predictionPtr,             simde_mm_unpacklo_epi64(xmm8, xmm8));
            simde_mm_storeu_si128((simde__m128i *)(predictionPtr + pStride), simde_mm_unpackhi_epi64(xmm8, xmm8));
            simde_mm_storeu_si128((simde__m128i *)(predictionPtr+2*pStride), simde_mm_unpacklo_epi64(xmm10, xmm10));
            simde_mm_storeu_si128((simde__m128i *)(predictionPtr+3*pStride), simde_mm_unpackhi_epi64(xmm10, xmm10));
            predictionPtr += (pStride << 2);                
            simde_mm_storeu_si128((simde__m128i *)predictionPtr,             simde_mm_unpacklo_epi64(xmm12, xmm12));
            simde_mm_storeu_si128((simde__m128i *)(predictionPtr + pStride), simde_mm_unpackhi_epi64(xmm12, xmm12));
            simde_mm_storeu_si128((simde__m128i *)(predictionPtr+2*pStride), simde_mm_unpacklo_epi64(xmm14, xmm14));
            simde_mm_storeu_si128((simde__m128i *)(predictionPtr+3*pStride), simde_mm_unpackhi_epi64(xmm14, xmm14));
        }
        else if (size == 8) {
            simde__m128i xmm0, xmm2, xmm4, xmm6;
            xmm0 = simde_mm_loadl_epi64((simde__m128i *)(refSamples + leftOffset));
            xmm0 = simde_mm_unpacklo_epi8(xmm0, xmm0);
            xmm4 = simde_mm_unpackhi_epi16(xmm0, xmm0);
            xmm0 = simde_mm_unpacklo_epi16(xmm0, xmm0);
            
            xmm2 = simde_mm_unpackhi_epi32(xmm0, xmm0);
            xmm0 = simde_mm_unpacklo_epi32(xmm0, xmm0);
            xmm6 = simde_mm_unpackhi_epi32(xmm4, xmm4);
            xmm4 = simde_mm_unpacklo_epi32(xmm4, xmm4);
            
            simde_mm_storel_epi64((simde__m128i *)(predictionPtr), xmm0);
            simde_mm_storel_epi64((simde__m128i *)(predictionPtr+pStride), simde_mm_srli_si128(xmm0, 8));
            simde_mm_storel_epi64((simde__m128i *)(predictionPtr+2*pStride), xmm2);
            simde_mm_storel_epi64((simde__m128i *)(predictionPtr+3*pStride), simde_mm_srli_si128(xmm2, 8));
            predictionPtr += (pStride << 2);
            simde_mm_storel_epi64((simde__m128i *)(predictionPtr), xmm4);
            simde_mm_storel_epi64((simde__m128i *)(predictionPtr+pStride), simde_mm_srli_si128(xmm4, 8));
            simde_mm_storel_epi64((simde__m128i *)(predictionPtr+2*pStride), xmm6);
            simde_mm_storel_epi64((simde__m128i *)(predictionPtr+3*pStride), simde_mm_srli_si128(xmm6, 8));
        }
        else {
            simde__m128i xmm0 = simde_mm_cvtsi32_si128(*(EB_U32*)(refSamples + leftOffset));
            xmm0 = simde_mm_unpacklo_epi8(xmm0, xmm0);
            xmm0 = simde_mm_unpacklo_epi16(xmm0, xmm0);
            *(EB_U32*)predictionPtr = simde_mm_cvtsi128_si32(xmm0);
            *(EB_U32*)(predictionPtr +pStride) = simde_mm_cvtsi128_si32(simde_mm_srli_si128(xmm0, 4));
            *(EB_U32*)(predictionPtr +2*pStride) = simde_mm_cvtsi128_si32(simde_mm_srli_si128(xmm0, 8));
            *(EB_U32*)(predictionPtr +3*pStride) = simde_mm_cvtsi128_si32(simde_mm_srli_si128(xmm0, 12));
        }
    }
    else {
        pStride <<= 1;
        simde__m128i xmm15 = simde_mm_set1_epi16(0x00FF);
        if (size == 32) {
            simde__m128i left0_14_even, left0_6_even, left8_14_even, left8_10, left12_14, left02, left46, skip_mask;
            skip_mask = simde_mm_set1_epi16(0x00FF);
            pStride <<= 1;
            MACRO_HORIZONTAL_LUMA_32X16(0)
            predictionPtr += (pStride << 2);
            MACRO_HORIZONTAL_LUMA_32X16(16)
        } else if (size == 16) {
            simde__m128i xmm0, xmm2, xmm4, xmm6;
            
            xmm0 = simde_mm_and_si128(simde_mm_loadu_si128((simde__m128i *)(refSamples + leftOffset)), xmm15);
            xmm0 = simde_mm_packus_epi16(xmm0, xmm0);
            xmm0 = simde_mm_unpacklo_epi8(xmm0, xmm0);
            xmm4 = simde_mm_unpackhi_epi16(xmm0, xmm0);
            xmm0 = simde_mm_unpacklo_epi16(xmm0, xmm0);
            
            xmm2 = simde_mm_unpackhi_epi32(xmm0, xmm0);
            xmm0 = simde_mm_unpacklo_epi32(xmm0, xmm0);
            xmm6 = simde_mm_unpackhi_epi32(xmm4, xmm4);
            xmm4 = simde_mm_unpacklo_epi32(xmm4, xmm4);

            simde_mm_storeu_si128((simde__m128i *)(predictionPtr),               simde_mm_unpacklo_epi64(xmm0, xmm0));
            simde_mm_storeu_si128((simde__m128i *)(predictionPtr + pStride),     simde_mm_unpackhi_epi64(xmm0, xmm0));
            simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 2 * pStride), simde_mm_unpacklo_epi64(xmm2, xmm2));
            simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 3 * pStride), simde_mm_unpackhi_epi64(xmm2, xmm2));
            predictionPtr += (pStride << 2);
            simde_mm_storeu_si128((simde__m128i *)(predictionPtr),               simde_mm_unpacklo_epi64(xmm4, xmm4));
            simde_mm_storeu_si128((simde__m128i *)(predictionPtr + pStride),     simde_mm_unpackhi_epi64(xmm4, xmm4));
            simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 2 * pStride), simde_mm_unpacklo_epi64(xmm6, xmm6));
            simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 3 * pStride), simde_mm_unpackhi_epi64(xmm6, xmm6));
        }
        else if (size == 8) {

            simde__m128i xmm2, xmm0;
            xmm0 = simde_mm_and_si128( simde_mm_loadl_epi64((simde__m128i *)(refSamples + leftOffset)), xmm15);
            xmm0 = simde_mm_packus_epi16(xmm0, xmm0);
            xmm0 = simde_mm_unpacklo_epi8(xmm0, xmm0);
            xmm0 = simde_mm_unpacklo_epi16(xmm0, xmm0);
            xmm2 = simde_mm_unpackhi_epi32(xmm0, xmm0);
            xmm0 = simde_mm_unpacklo_epi32(xmm0, xmm0);
            
            simde_mm_storel_epi64((simde__m128i *)(predictionPtr), xmm0);
            simde_mm_storel_epi64((simde__m128i *)(predictionPtr + pStride), simde_mm_srli_si128(xmm0, 8));
            simde_mm_storel_epi64((simde__m128i *)(predictionPtr + 2*pStride), xmm2);
            simde_mm_storel_epi64((simde__m128i *)(predictionPtr + 3*pStride), simde_mm_srli_si128(xmm2, 8));
        }
        else {
            simde__m128i xmm0;
            xmm0 = simde_mm_and_si128(simde_mm_cvtsi32_si128(*(EB_U32*)(refSamples + leftOffset)), xmm15);
            xmm0 = simde_mm_packus_epi16(xmm0, xmm0);
            xmm0 = simde_mm_unpacklo_epi8(xmm0, xmm0);
            xmm0 = simde_mm_unpacklo_epi16(xmm0, xmm0);

            *(EB_U32*)(predictionPtr) = simde_mm_cvtsi128_si32(xmm0);
            *(EB_U32*)(predictionPtr + pStride) = simde_mm_cvtsi128_si32(simde_mm_srli_si128(xmm0, 4));
        }     
    }
}

void IntraModePlanar16bit_SSE2_INTRIN(
    const EB_U32   size,                       //input parameter, denotes the size of the current PU
    EB_U16         *refSamples,                 //input parameter, pointer to the reference samples
    EB_U16         *predictionPtr,              //output parameter, pointer to the prediction
    const EB_U32   predictionBufferStride,     //input parameter, denotes the stride for the prediction ptr
    const EB_BOOL  skip)                       //skip half rows
{
    EB_U32 topOffset = (size << 1) + 1;
    EB_U32 leftOffset = 0;
    EB_U32 bottomLeftOffset = leftOffset + size;
    EB_U32 topRightOffset = topOffset + size;
    EB_U32 pStride = predictionBufferStride;
    EB_U32 count, reverseCnt, coefOffset;
    simde__m128i  leftCoeff, pred, left, left16, top, topRight, topRightAddSize, bottomLeft, bottomLeftTotal, bottomLeftTotal16;
    EB_ALIGN(16) const EB_S16 * coeffArray = IntraPredictionConst_SSE2;

    if (size != 4) {
        simde__m128i xmm_TopRight, xmm_BottomLeft;
        simde__m128i pred_0, pred_1, pred_2, pred_3;
        
        xmm_TopRight = simde_mm_cvtsi32_si128((EB_U32)*(refSamples + topRightOffset));
        xmm_TopRight = simde_mm_unpacklo_epi16(xmm_TopRight, xmm_TopRight);
        xmm_TopRight = simde_mm_unpacklo_epi32(xmm_TopRight, xmm_TopRight);
        xmm_TopRight = simde_mm_unpacklo_epi64(xmm_TopRight, xmm_TopRight);
        xmm_BottomLeft = simde_mm_cvtsi32_si128((EB_U32)*(refSamples + bottomLeftOffset));
        xmm_BottomLeft = simde_mm_unpacklo_epi16(xmm_BottomLeft, xmm_BottomLeft);
        xmm_BottomLeft = simde_mm_unpacklo_epi32(xmm_BottomLeft, xmm_BottomLeft);
        xmm_BottomLeft = simde_mm_unpacklo_epi64(xmm_BottomLeft, xmm_BottomLeft);
        
        if (size == 32) {
            simde__m128i xmm_ref, xmm_ref16;
            simde__m128i xmm_topRightAddSize0, xmm_topRightAddSize1, xmm_topRightAddSize2, xmm_topRightAddSize3, xmm_top0, xmm_top1, xmm_top2, xmm_top3;

            coefOffset = OFFSET_31; // The coefficients will be size-1, size-2, ... so we start at 31, 30, ... and reverse towards 0

            xmm_topRightAddSize0 = simde_mm_add_epi16(simde_mm_mullo_epi16(xmm_TopRight, simde_mm_load_si128((simde__m128i *)(coeffArray + OFFSET_1TO8))), simde_mm_load_si128((simde__m128i *)(coeffArray + OFFSET_32)));
            xmm_topRightAddSize1 = simde_mm_add_epi16(simde_mm_mullo_epi16(xmm_TopRight, simde_mm_load_si128((simde__m128i *)(coeffArray + OFFSET_9TO16))), simde_mm_load_si128((simde__m128i *)(coeffArray + OFFSET_32)));
            xmm_topRightAddSize2 = simde_mm_add_epi16(simde_mm_mullo_epi16(xmm_TopRight, simde_mm_load_si128((simde__m128i *)(coeffArray + OFFSET_17TO24))), simde_mm_load_si128((simde__m128i *)(coeffArray + OFFSET_32)));
            xmm_topRightAddSize3 = simde_mm_add_epi16(simde_mm_mullo_epi16(xmm_TopRight, simde_mm_load_si128((simde__m128i *)(coeffArray + OFFSET_25TO32))), simde_mm_load_si128((simde__m128i *)(coeffArray + OFFSET_32)));

            xmm_top0 = simde_mm_loadu_si128((simde__m128i *)(refSamples + topOffset));
            xmm_top1 = simde_mm_loadu_si128((simde__m128i *)(refSamples + topOffset + 8));
            xmm_top2 = simde_mm_loadu_si128((simde__m128i *)(refSamples + topOffset + 16));
            xmm_top3 = simde_mm_loadu_si128((simde__m128i *)(refSamples + topOffset + 24));
            refSamples += leftOffset;

            if (!skip) {
                EB_U32 count1; 
                
                // The coefficients will be size-1, size-2, ... so we start at 31, 30, ... and reverse towards 0
                for (reverseCnt = 8; reverseCnt > 0; reverseCnt -= 2) {
                
                    bottomLeftTotal = simde_mm_mullo_epi16(xmm_BottomLeft, simde_mm_load_si128((simde__m128i *)(coeffArray + 4 * reverseCnt + OFFSET_1TO8 - 32)));
                    
                    xmm_ref = simde_mm_loadu_si128((simde__m128i *)(refSamples));
                    refSamples += 8;

                    for (count1 = 0; count1 < 8; ++count1) {
                        
                        xmm_ref16 = simde_mm_unpacklo_epi16(xmm_ref, xmm_ref);
                        xmm_ref16 = simde_mm_unpacklo_epi32(xmm_ref16, xmm_ref16);
                        xmm_ref16 = simde_mm_unpacklo_epi64(xmm_ref16, xmm_ref16);
                        
                        bottomLeftTotal16 = simde_mm_unpacklo_epi16(bottomLeftTotal, bottomLeftTotal);
                        bottomLeftTotal16 = simde_mm_unpacklo_epi32(bottomLeftTotal16, bottomLeftTotal16);
                        bottomLeftTotal16 = simde_mm_unpacklo_epi64(bottomLeftTotal16, bottomLeftTotal16);                        
                        
                        pred_0 =  simde_mm_srli_epi16(simde_mm_add_epi16(simde_mm_add_epi16(simde_mm_add_epi16(simde_mm_mullo_epi16(xmm_ref16, simde_mm_load_si128((simde__m128i *)(coeffArray + OFFSET_31TO24))), xmm_topRightAddSize0), bottomLeftTotal16), simde_mm_mullo_epi16(xmm_top0, simde_mm_load_si128((simde__m128i *)(coeffArray + coefOffset)))), 6);
                        pred_1 =  simde_mm_srli_epi16(simde_mm_add_epi16(simde_mm_add_epi16(simde_mm_add_epi16(simde_mm_mullo_epi16(xmm_ref16, simde_mm_load_si128((simde__m128i *)(coeffArray + OFFSET_23TO16))), xmm_topRightAddSize1), bottomLeftTotal16), simde_mm_mullo_epi16(xmm_top1, simde_mm_load_si128((simde__m128i *)(coeffArray + coefOffset)))), 6);
                        pred_2 = simde_mm_srli_epi16(simde_mm_add_epi16(simde_mm_add_epi16(simde_mm_add_epi16(simde_mm_mullo_epi16(xmm_ref16, simde_mm_load_si128((simde__m128i *)(coeffArray + OFFSET_15TO8))), xmm_topRightAddSize2), bottomLeftTotal16), simde_mm_mullo_epi16(xmm_top2, simde_mm_load_si128((simde__m128i *)(coeffArray + coefOffset)))), 6);
                        pred_3 = simde_mm_srli_epi16(simde_mm_add_epi16(simde_mm_add_epi16(simde_mm_add_epi16(simde_mm_mullo_epi16(xmm_ref16, simde_mm_load_si128((simde__m128i *)(coeffArray + OFFSET_7TO0))), xmm_topRightAddSize3), bottomLeftTotal16), simde_mm_mullo_epi16(xmm_top3, simde_mm_load_si128((simde__m128i *)(coeffArray + coefOffset)))), 6);
                        
                        simde_mm_storeu_si128((simde__m128i *)(predictionPtr), pred_0);
                        simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 8), pred_1);
                        simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 16), pred_2);
                        simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 24), pred_3);
                        
                        // For next iteration
                        coefOffset -= 8;
                        predictionPtr += pStride;
                        xmm_ref = simde_mm_srli_si128(xmm_ref, 2);
                        bottomLeftTotal = simde_mm_srli_si128(bottomLeftTotal, 2);
                    }
                }
            }
            else {
                for (reverseCnt = 4; reverseCnt > 0; reverseCnt -= 2) {

                    bottomLeftTotal = simde_mm_mullo_epi16(xmm_BottomLeft, simde_mm_load_si128((simde__m128i *)(coeffArray + 4 * reverseCnt + OFFSET_1TO15 - 16)));
                    xmm_ref = simde_mm_packs_epi32(simde_mm_srli_epi32(simde_mm_slli_epi32(simde_mm_loadu_si128((simde__m128i *)(refSamples)), 16), 16),           
                                              simde_mm_srli_epi32(simde_mm_slli_epi32(simde_mm_loadu_si128((simde__m128i *)(refSamples + 8)), 16), 16));
                    refSamples += 16;

                    for (count = 0; count < 8; ++count) {
                        
                        xmm_ref16 = simde_mm_unpacklo_epi16(xmm_ref, xmm_ref);
                        xmm_ref16 = simde_mm_unpacklo_epi32(xmm_ref16, xmm_ref16);
                        xmm_ref16 = simde_mm_unpacklo_epi64(xmm_ref16, xmm_ref16);
                        
                        bottomLeftTotal16 = simde_mm_unpacklo_epi16(bottomLeftTotal, bottomLeftTotal);
                        bottomLeftTotal16 = simde_mm_unpacklo_epi32(bottomLeftTotal16, bottomLeftTotal16);
                        bottomLeftTotal16 = simde_mm_unpacklo_epi64(bottomLeftTotal16, bottomLeftTotal16);
                        
                        pred_0 =  simde_mm_srli_epi16(simde_mm_add_epi16(simde_mm_add_epi16(simde_mm_add_epi16(simde_mm_mullo_epi16(xmm_ref16, simde_mm_load_si128((simde__m128i *)(coeffArray + OFFSET_31TO24))), xmm_topRightAddSize0), bottomLeftTotal16), simde_mm_mullo_epi16(xmm_top0, simde_mm_load_si128((simde__m128i *)(coeffArray + coefOffset)))),6);
                        pred_1 =  simde_mm_srli_epi16(simde_mm_add_epi16(simde_mm_add_epi16(simde_mm_add_epi16(simde_mm_mullo_epi16(xmm_ref16, simde_mm_load_si128((simde__m128i *)(coeffArray + OFFSET_23TO16))), xmm_topRightAddSize1), bottomLeftTotal16), simde_mm_mullo_epi16(xmm_top1, simde_mm_load_si128((simde__m128i *)(coeffArray + coefOffset)))),6);
                        pred_2 = simde_mm_srli_epi16(simde_mm_add_epi16(simde_mm_add_epi16(simde_mm_add_epi16(simde_mm_mullo_epi16(xmm_ref16, simde_mm_load_si128((simde__m128i *)(coeffArray + OFFSET_15TO8))), xmm_topRightAddSize2), bottomLeftTotal16), simde_mm_mullo_epi16(xmm_top2, simde_mm_load_si128((simde__m128i *)(coeffArray + coefOffset)))),6);
                        pred_3 = simde_mm_srli_epi16(simde_mm_add_epi16(simde_mm_add_epi16(simde_mm_add_epi16(simde_mm_mullo_epi16(xmm_ref16, simde_mm_load_si128((simde__m128i *)(coeffArray + OFFSET_7TO0))), xmm_topRightAddSize3), bottomLeftTotal16), simde_mm_mullo_epi16(xmm_top3, simde_mm_load_si128((simde__m128i *)(coeffArray + coefOffset)))),6);
                        
                        simde_mm_storeu_si128((simde__m128i *)(predictionPtr), pred_0);
                        simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 8), pred_1);
                        simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 16), pred_2);
                        simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 24), pred_3);

                        // For next iteration
                        coefOffset -= 16;
                        predictionPtr += 2 * pStride;
                        bottomLeftTotal = simde_mm_srli_si128(bottomLeftTotal, 2);
                        xmm_ref = simde_mm_srli_si128(xmm_ref, 2);
                    }
                }
            }
        }
        else if (size == 16) {
            simde__m128i topRightTotal0, topRightTotal1, top0, top1, left1, bottomLeftTotal1, leftCoeff0, leftCoeff1;

            coefOffset = OFFSET_15; // The coefficients will be size-1, size-2, ... so we start at 15, 14, ... and reverse towards 0
            
            topRightTotal0 = simde_mm_add_epi16(simde_mm_mullo_epi16(xmm_TopRight, simde_mm_load_si128((simde__m128i *)(coeffArray + OFFSET_1TO8))), simde_mm_load_si128((simde__m128i *)(coeffArray + OFFSET_16)));
            topRightTotal1 = simde_mm_add_epi16(simde_mm_mullo_epi16(xmm_TopRight, simde_mm_load_si128((simde__m128i *)(coeffArray + OFFSET_9TO16))), simde_mm_load_si128((simde__m128i *)(coeffArray + OFFSET_16)));
            top0 = simde_mm_loadu_si128((simde__m128i *)(refSamples + topOffset));
            top1 = simde_mm_loadu_si128((simde__m128i *)(refSamples + topOffset + 8));
            left = simde_mm_loadu_si128((simde__m128i *)(refSamples + leftOffset));
            left1 = simde_mm_loadu_si128((simde__m128i *)(refSamples + leftOffset + 8));
            leftCoeff0 = simde_mm_load_si128((simde__m128i *)(coeffArray + OFFSET_15TO8));
            leftCoeff1 = simde_mm_load_si128((simde__m128i *)(coeffArray + OFFSET_7TO0));

            if (!skip) {
                EB_U64 count1;
                bottomLeftTotal = simde_mm_mullo_epi16(xmm_BottomLeft, simde_mm_load_si128((simde__m128i *)(coeffArray + OFFSET_1TO8)));
                bottomLeftTotal1 = simde_mm_mullo_epi16(xmm_BottomLeft, simde_mm_load_si128((simde__m128i *)(coeffArray + OFFSET_9TO16)));

                for (count = 0; count < 2; ++count) {
                    for (count1 = 0; count1 < 8; ++count1) {

                        left16 = simde_mm_unpacklo_epi16(left, left);
                        left16 = simde_mm_unpacklo_epi32(left16, left16);
                        left16 = simde_mm_unpacklo_epi64(left16, left16);

                        bottomLeftTotal16 = simde_mm_unpacklo_epi16(bottomLeftTotal, bottomLeftTotal);
                        bottomLeftTotal16 = simde_mm_unpacklo_epi32(bottomLeftTotal16, bottomLeftTotal16);
                        bottomLeftTotal16 = simde_mm_unpacklo_epi64(bottomLeftTotal16, bottomLeftTotal16);                        
                        
                        pred_0 = simde_mm_srli_epi16(simde_mm_add_epi16(simde_mm_add_epi16(simde_mm_add_epi16(simde_mm_mullo_epi16(left16, leftCoeff0), topRightTotal0), bottomLeftTotal16), simde_mm_mullo_epi16(top0, simde_mm_load_si128((simde__m128i *)(coeffArray + coefOffset)))), 5);
                        pred_1 = simde_mm_srli_epi16(simde_mm_add_epi16(simde_mm_add_epi16(simde_mm_add_epi16(simde_mm_mullo_epi16(left16, leftCoeff1), topRightTotal1), bottomLeftTotal16), simde_mm_mullo_epi16(top1, simde_mm_load_si128((simde__m128i *)(coeffArray + coefOffset)))), 5);

                        simde_mm_storeu_si128((simde__m128i *)predictionPtr, pred_0);
                        simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 8), pred_1);
                        
                        // Next iteration
                        coefOffset -= 8;
                        predictionPtr += pStride;
                        left = simde_mm_srli_si128(left, 2);
                        bottomLeftTotal = simde_mm_srli_si128(bottomLeftTotal, 2);
                    }
                    bottomLeftTotal = bottomLeftTotal1;
                    left = left1;
                }
            }
            else {
                bottomLeftTotal = simde_mm_mullo_epi16(xmm_BottomLeft, simde_mm_load_si128((simde__m128i *)(coeffArray + OFFSET_1TO15)));                
                left = simde_mm_packs_epi32(simde_mm_srli_epi32(simde_mm_slli_epi32(left, 16), 16), simde_mm_srli_epi32(simde_mm_slli_epi32(left1, 16), 16));

                for (count = 0; count < 8; ++count) {
                    
                    left16 = simde_mm_unpacklo_epi16(left, left);
                    left16 = simde_mm_unpacklo_epi32(left16, left16);
                    left16 = simde_mm_unpacklo_epi64(left16, left16);                    
                    
                    bottomLeftTotal16 = simde_mm_unpacklo_epi16(bottomLeftTotal, bottomLeftTotal);
                    bottomLeftTotal16 = simde_mm_unpacklo_epi32(bottomLeftTotal16, bottomLeftTotal16);
                    bottomLeftTotal16 = simde_mm_unpacklo_epi64(bottomLeftTotal16, bottomLeftTotal16);                          
                    
                    pred_0 = simde_mm_srli_epi16(simde_mm_add_epi16(simde_mm_add_epi16(simde_mm_add_epi16(simde_mm_mullo_epi16(left16, leftCoeff0), topRightTotal0), bottomLeftTotal16), simde_mm_mullo_epi16(top0, simde_mm_load_si128((simde__m128i *)(coeffArray + coefOffset)))), 5);
                    pred_1 = simde_mm_srli_epi16(simde_mm_add_epi16(simde_mm_add_epi16(simde_mm_add_epi16(simde_mm_mullo_epi16(left16, leftCoeff1), topRightTotal1), bottomLeftTotal16), simde_mm_mullo_epi16(top1, simde_mm_load_si128((simde__m128i *)(coeffArray + coefOffset)))), 5);
                    
                    simde_mm_storeu_si128((simde__m128i *)predictionPtr, pred_0);
                    simde_mm_storeu_si128((simde__m128i *)(predictionPtr + 8), pred_1);

                    // Next iteration
                    coefOffset -= 16;
                    predictionPtr += 2 * pStride;
                    bottomLeftTotal = simde_mm_srli_si128(bottomLeftTotal, 2);
                    left = simde_mm_srli_si128(left, 2);
                }
            }
        }
        else {
            
            topRightAddSize = simde_mm_add_epi16(simde_mm_mullo_epi16(xmm_TopRight, simde_mm_load_si128((simde__m128i *)(coeffArray + OFFSET_1TO8))), simde_mm_load_si128((simde__m128i *)(coeffArray + OFFSET_8)));
            top = simde_mm_loadu_si128((simde__m128i *)(refSamples + topOffset));
            left = simde_mm_loadu_si128((simde__m128i *)(refSamples + leftOffset));
            leftCoeff = simde_mm_load_si128((simde__m128i *)(coeffArray + OFFSET_7TO0));

            if (!skip) {
                
                bottomLeftTotal = simde_mm_mullo_epi16(xmm_BottomLeft, simde_mm_load_si128((simde__m128i *)(coeffArray + OFFSET_1TO8)));
                for (count = 0; count < 8; ++count) {
                    
                    left16 = simde_mm_unpacklo_epi16(left, left);
                    left16 = simde_mm_unpacklo_epi32(left16, left16);
                    left16 = simde_mm_unpacklo_epi64(left16, left16);
                    
                    bottomLeftTotal16 = simde_mm_unpacklo_epi16(bottomLeftTotal, bottomLeftTotal);
                    bottomLeftTotal16 = simde_mm_unpacklo_epi32(bottomLeftTotal16, bottomLeftTotal16);
                    bottomLeftTotal16 = simde_mm_unpacklo_epi64(bottomLeftTotal16, bottomLeftTotal16);
                    
                    pred = simde_mm_srli_epi16(simde_mm_add_epi16(simde_mm_add_epi16(simde_mm_add_epi16(simde_mm_mullo_epi16(left16, leftCoeff), topRightAddSize), bottomLeftTotal16), simde_mm_mullo_epi16(top, simde_mm_load_si128((simde__m128i *)(coeffArray + OFFSET_7)))), 4);
                    simde_mm_storeu_si128((simde__m128i *)(predictionPtr), pred);

                    // Next iteration
                    coeffArray -= 8;
                    predictionPtr += pStride;
                    left = simde_mm_srli_si128(left, 2);
                    bottomLeftTotal = simde_mm_srli_si128(bottomLeftTotal, 2);
                } 
            } 
            else {
                bottomLeftTotal = simde_mm_mullo_epi16(xmm_BottomLeft, simde_mm_load_si128((simde__m128i *)(coeffArray + OFFSET_1TO15)));
                left = simde_mm_srli_epi32(simde_mm_slli_epi32(left, 16), 16);

                for (count = 0; count < 4; ++count) {
                    
                    left16 = simde_mm_unpacklo_epi16(left, left);
                    left16 = simde_mm_unpacklo_epi32(left16, left16);
                    left16 = simde_mm_unpacklo_epi64(left16, left16);
                    
                    bottomLeftTotal16 = simde_mm_unpacklo_epi16(bottomLeftTotal, bottomLeftTotal);
                    bottomLeftTotal16 = simde_mm_unpacklo_epi32(bottomLeftTotal16, bottomLeftTotal16);
                    bottomLeftTotal16 = simde_mm_unpacklo_epi64(bottomLeftTotal16, bottomLeftTotal16);

                    pred = simde_mm_srli_epi16(simde_mm_add_epi16(simde_mm_add_epi16(simde_mm_add_epi16(simde_mm_mullo_epi16(left16, leftCoeff), topRightAddSize), bottomLeftTotal16), simde_mm_mullo_epi16(top, simde_mm_load_si128((simde__m128i *)(coeffArray + OFFSET_7)))), 4);
                    simde_mm_storeu_si128((simde__m128i *)predictionPtr, pred);

                    // Next iteration
                    coeffArray -= 16;
                    predictionPtr += 2 * pStride;
                    left = simde_mm_srli_si128(left, 4);
                    bottomLeftTotal = simde_mm_srli_si128(bottomLeftTotal, 2);
                }
            }
        }
    }
    else {
        
        topRight = simde_mm_cvtsi32_si128((EB_U32)*(refSamples + topRightOffset));
        topRight = simde_mm_unpacklo_epi16(topRight, topRight);
        topRight = simde_mm_unpacklo_epi32(topRight, topRight);

        bottomLeft = simde_mm_cvtsi32_si128((EB_U32)*(refSamples + bottomLeftOffset));
        bottomLeft = simde_mm_unpacklo_epi16(bottomLeft, bottomLeft);
        bottomLeft = simde_mm_unpacklo_epi32(bottomLeft, bottomLeft);
        
        topRightAddSize = simde_mm_add_epi16(simde_mm_mullo_epi16(topRight, simde_mm_load_si128((simde__m128i *)(coeffArray + OFFSET_1TO8))), simde_mm_load_si128((simde__m128i *)(coeffArray + OFFSET_4)));
        
        top = simde_mm_loadl_epi64((simde__m128i *)(refSamples + topOffset));
        left = simde_mm_loadl_epi64((simde__m128i *)(refSamples + leftOffset));
        leftCoeff = simde_mm_loadl_epi64((simde__m128i *)(coeffArray + OFFSET_3TO0));

        if (!skip) {
            bottomLeftTotal = simde_mm_mullo_epi16(bottomLeft, simde_mm_load_si128((simde__m128i *)(coeffArray + OFFSET_1TO8)));
            
            for (count = 0; count < 4; ++count) {
                
                left16 = simde_mm_unpacklo_epi16(left, left);
                left16 = simde_mm_unpacklo_epi32(left16, left16);
                
                bottomLeftTotal16 = simde_mm_unpacklo_epi16(bottomLeftTotal, bottomLeftTotal);
                bottomLeftTotal16 = simde_mm_unpacklo_epi32(bottomLeftTotal16, bottomLeftTotal16);

                pred = simde_mm_srli_epi16(simde_mm_add_epi16(simde_mm_add_epi16(simde_mm_add_epi16(simde_mm_mullo_epi16(left16, leftCoeff), topRightAddSize), bottomLeftTotal16), simde_mm_mullo_epi16(top, simde_mm_load_si128((simde__m128i *)(coeffArray + OFFSET_3)))), 3);
                simde_mm_storel_epi64((simde__m128i *)(predictionPtr), pred);

                // Next iteration
                coeffArray -= 8;
                predictionPtr += pStride;
                bottomLeftTotal = simde_mm_srli_epi64(bottomLeftTotal, 16);
                left = simde_mm_srli_epi64(left, 16);
            }
        }
        else {
            bottomLeftTotal = simde_mm_mullo_epi16(bottomLeft, simde_mm_load_si128((simde__m128i *)(coeffArray + OFFSET_1TO15)));
            left = simde_mm_srli_epi32(simde_mm_slli_epi32(left, 16), 16);

            for (count = 0; count < 2; ++count) {
                
                left16 = simde_mm_unpacklo_epi16(left, left);
                left16 = simde_mm_unpacklo_epi32(left16, left16);
                
                bottomLeftTotal16 = simde_mm_unpacklo_epi16(bottomLeftTotal, bottomLeftTotal);
                bottomLeftTotal16 = simde_mm_unpacklo_epi32(bottomLeftTotal16, bottomLeftTotal16);
                
                pred = simde_mm_srli_epi16(simde_mm_add_epi16(simde_mm_add_epi16(simde_mm_add_epi16(simde_mm_mullo_epi16(left16, leftCoeff), topRightAddSize), bottomLeftTotal16), simde_mm_mullo_epi16(top, simde_mm_load_si128((simde__m128i *)(coeffArray + OFFSET_3)))), 3);
                simde_mm_storel_epi64((simde__m128i *)predictionPtr, pred);

                // Next iteration
                coeffArray -= 16;
                predictionPtr += 2 * pStride;
                left = simde_mm_srli_epi64(left, 32);
                bottomLeftTotal = simde_mm_srli_epi64(bottomLeftTotal, 16);
            }
        }
    }
}
