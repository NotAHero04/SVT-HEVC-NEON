/*
* Copyright(c) 2018 Intel Corporation
* SPDX - License - Identifier: BSD - 2 - Clause - Patent
*/

#include "../../../simde/simde/x86/sse2.h"
#include "EbDefinitions.h"

#include "EbDeblockingFilter_SSE2.h"

void Chroma2SampleEdgeDLFCore16bit_SSE2_INTRIN(
    EB_U16				  *edgeStartSampleCb,
    EB_U16				  *edgeStartSampleCr,
    EB_U32                 reconChromaPicStride,
    EB_BOOL                isVerticalEdge,
    EB_U8                  cbTc,
    EB_U8                  crTc)
{    
    simde__m128i xmm_tcN = simde_mm_setzero_si128();
    simde__m128i xmm0 = simde_mm_setzero_si128();
    simde__m128i xmm_4 = simde_mm_set1_epi16(4);
    simde__m128i xmm_Max10bit = simde_mm_set1_epi16(0x03FF);
    simde__m128i xmm_p0, xmm_p1, xmm_q0, xmm_q1, xmm_p0Cr, xmm_temp1, xmm_delta;
    simde__m128i xmm_Tc = simde_mm_unpacklo_epi8(simde_mm_cvtsi32_si128(cbTc), simde_mm_cvtsi32_si128(crTc));
    xmm_Tc = simde_mm_unpacklo_epi8(xmm_Tc, xmm_Tc);
    xmm_Tc = simde_mm_unpacklo_epi8(xmm_Tc, xmm0);
    xmm_Tc = simde_mm_unpacklo_epi64(xmm_Tc, xmm_Tc);
    xmm_tcN = simde_mm_sub_epi16(xmm_tcN, xmm_Tc);

    if (0 == isVerticalEdge) {

        xmm_p0 = simde_mm_unpacklo_epi32(simde_mm_loadl_epi64((simde__m128i *)(edgeStartSampleCb -  reconChromaPicStride)), simde_mm_loadl_epi64((simde__m128i *)(edgeStartSampleCr - reconChromaPicStride)));
        xmm_p1 = simde_mm_unpacklo_epi32(simde_mm_loadl_epi64((simde__m128i *)(edgeStartSampleCb -  (reconChromaPicStride << 1))), simde_mm_loadl_epi64((simde__m128i *)(edgeStartSampleCr - (reconChromaPicStride << 1))));
        xmm_q0 = simde_mm_unpacklo_epi32(simde_mm_loadl_epi64((simde__m128i *)(edgeStartSampleCb)), simde_mm_loadl_epi64((simde__m128i *)(edgeStartSampleCr)));
        xmm_q1 = simde_mm_unpacklo_epi32(simde_mm_loadl_epi64((simde__m128i *)(edgeStartSampleCb +  reconChromaPicStride)), simde_mm_loadl_epi64((simde__m128i *)(edgeStartSampleCr + reconChromaPicStride)));
                
        xmm_temp1 = simde_mm_srai_epi16(simde_mm_add_epi16(simde_mm_slli_epi16(simde_mm_sub_epi16(xmm_q0, xmm_p0), 2),  simde_mm_add_epi16(simde_mm_sub_epi16(xmm_p1, xmm_q1), xmm_4)), 3);
        xmm_delta = simde_mm_max_epi16(xmm_temp1, xmm_tcN);
        xmm_delta = simde_mm_min_epi16(xmm_delta, xmm_Tc);

        xmm_p0 = simde_mm_unpacklo_epi64(simde_mm_add_epi16(xmm_p0, xmm_delta), simde_mm_sub_epi16(xmm_q0, xmm_delta));
        xmm_p0 = simde_mm_max_epi16(simde_mm_min_epi16(xmm_p0, xmm_Max10bit), xmm0);

        *(EB_U32 *)(edgeStartSampleCb - reconChromaPicStride) = simde_mm_cvtsi128_si32(xmm_p0);
        *(EB_U32 *)(edgeStartSampleCr - reconChromaPicStride) = simde_mm_cvtsi128_si32(simde_mm_srli_si128(xmm_p0, 4));
        *(EB_U32 *)(edgeStartSampleCb) = simde_mm_cvtsi128_si32( simde_mm_srli_si128(xmm_p0, 8));
        *(EB_U32 *)(edgeStartSampleCr) = simde_mm_cvtsi128_si32(simde_mm_srli_si128(xmm_p0, 12));
    }
    else {
        xmm_p0 = simde_mm_unpacklo_epi16(simde_mm_loadl_epi64((simde__m128i *)(edgeStartSampleCb - 2)), simde_mm_loadl_epi64((simde__m128i *)(edgeStartSampleCb + reconChromaPicStride - 2)));       
        xmm_p0Cr = simde_mm_unpacklo_epi16(simde_mm_loadl_epi64((simde__m128i *)(edgeStartSampleCr - 2)), simde_mm_loadl_epi64((simde__m128i *)(edgeStartSampleCr + reconChromaPicStride - 2))); 
  
        xmm_q0 = simde_mm_unpackhi_epi32(xmm_p0, xmm_p0Cr); 
        xmm_p0 = simde_mm_unpacklo_epi32(xmm_p0, xmm_p0Cr);                       
        xmm_p0 = simde_mm_unpacklo_epi64(simde_mm_srli_si128(xmm_p0, 8), xmm_p0);    
                                   
        xmm_temp1 = simde_mm_sub_epi16(xmm_q0, xmm_p0);     
        xmm_temp1 = simde_mm_srai_epi16(simde_mm_add_epi16(simde_mm_sub_epi16(simde_mm_slli_epi16(xmm_temp1, 2), simde_mm_srli_si128(xmm_temp1, 8)), xmm_4), 3);           
        xmm_delta = simde_mm_min_epi16(simde_mm_max_epi16(xmm_temp1, xmm_tcN), xmm_Tc);

        xmm_p0 = simde_mm_unpacklo_epi16(simde_mm_add_epi16(xmm_p0, xmm_delta), simde_mm_sub_epi16(xmm_q0, xmm_delta)); 
        xmm_p0 = simde_mm_max_epi16(simde_mm_min_epi16(xmm_p0, xmm_Max10bit), xmm0); 

        *(EB_U32*)(edgeStartSampleCb-1) = simde_mm_cvtsi128_si32(xmm_p0);
        *(EB_U32*)(edgeStartSampleCb+reconChromaPicStride-1) = simde_mm_cvtsi128_si32(simde_mm_srli_si128(xmm_p0, 4));
        *(EB_U32*)(edgeStartSampleCr-1) = simde_mm_cvtsi128_si32(simde_mm_srli_si128(xmm_p0, 8));
        *(EB_U32*)(edgeStartSampleCr+reconChromaPicStride-1) = simde_mm_cvtsi128_si32(simde_mm_srli_si128(xmm_p0, 12));
    }

    return;
}
