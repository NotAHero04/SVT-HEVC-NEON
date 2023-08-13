/*
* Copyright(c) 2018 Intel Corporation
* SPDX - License - Identifier: BSD - 2 - Clause - Patent
*/


#include "EbCombinedAveragingSAD_Intrinsic_AVX2.h"
#include "../../../simde/simde/x86/avx2.h"

#define simde_mm256_set_m128i(/* simde__m128i */ hi, /* simde__m128i */ lo) \
    simde_mm256_insertf128_si256(simde_mm256_castsi128_si256(lo), (hi), 0x1)

EB_U64 ComputeMean8x8_AVX2_INTRIN(
    EB_U8 *  inputSamples,      // input parameter, input samples Ptr
    EB_U32   inputStride,       // input parameter, input stride
    EB_U32   inputAreaWidth,    // input parameter, input area width
    EB_U32   inputAreaHeight)   // input parameter, input area height
{
	simde__m256i sum,sum2 ,xmm2, xmm1, sum1, xmm0 = simde_mm256_setzero_si256();
	simde__m128i  upper, lower, mean = simde_mm_setzero_si128() ;
	EB_U64 result;
	xmm1=simde_mm256_sad_epu8( xmm0 ,simde_mm256_set_m128i( simde_mm_loadl_epi64((simde__m128i *)(inputSamples+inputStride)) , simde_mm_loadl_epi64((simde__m128i *)(inputSamples)) ));
	xmm2= simde_mm256_sad_epu8(xmm0,simde_mm256_set_m128i(simde_mm_loadl_epi64((simde__m128i *)(inputSamples+3*inputStride)) ,simde_mm_loadl_epi64((simde__m128i *)(inputSamples+2*inputStride)) ) ) ;
	sum1 = simde_mm256_add_epi16(xmm1, xmm2);
	
	inputSamples += 4 * inputStride;
	
	xmm1= simde_mm256_sad_epu8(xmm0,simde_mm256_set_m128i( simde_mm_loadl_epi64((simde__m128i *)(inputSamples+inputStride)) , simde_mm_loadl_epi64((simde__m128i *)(inputSamples)) )) ;
	xmm2= simde_mm256_sad_epu8(xmm0, simde_mm256_set_m128i(simde_mm_loadl_epi64((simde__m128i *)(inputSamples+3*inputStride)) ,simde_mm_loadl_epi64((simde__m128i *)(inputSamples+2*inputStride)) ) );
	sum2 = simde_mm256_add_epi16(xmm1, xmm2);
	
    sum = simde_mm256_add_epi16(sum1, sum2);
	upper = simde_mm256_extractf128_si256(sum,1) ; //extract upper 128 bit
	upper = simde_mm_add_epi32(upper, simde_mm_srli_si128(upper, 8)); // shift 2nd 16 bits to the 1st and sum both
	
	lower = simde_mm256_extractf128_si256(sum,0) ; //extract lower 128 bit
	lower = simde_mm_add_epi32(lower, simde_mm_srli_si128(lower, 8)); // shift 2nd 16 bits to the 1st and sum both
	
	mean = simde_mm_add_epi32(lower,upper);
	
	(void)inputAreaWidth;
    (void)inputAreaHeight;
    
    result = (EB_U64)simde_mm_cvtsi128_si32(mean) << 2;
    return result;
	
} 

/********************************************************************************************************************************/
    void  ComputeIntermVarFour8x8_AVX2_INTRIN(
        EB_U8 *  inputSamples,
        EB_U16   inputStride,
        EB_U64 * meanOf8x8Blocks,      // mean of four  8x8
        EB_U64 * meanOfSquared8x8Blocks)  // meanSquared
    {

        simde__m256i ymm1, ymm2, ymm3, ymm4, ymm_sum1, ymm_sum2, ymm_FinalSum,ymm_shift,/* ymm_blockMeanSquared*///,
                ymm_in,ymm_in_2S,ymm_in_second,ymm_in_2S_second,ymm_shiftSquared,ymm_permute8,
                ymm_result,ymm_blockMeanSquaredlow,ymm_blockMeanSquaredHi,ymm_inputlo,ymm_inputhi;
        
        simde__m128i ymm_blockMeanSquaredlo,ymm_blockMeanSquaredhi,ymm_resultlo,ymm_resulthi;
       
        simde__m256i ymm_zero = simde_mm256_setzero_si256();
        simde__m128i xmm_zero = simde_mm_setzero_si128();

        ymm_in    = simde_mm256_loadu_si256((simde__m256i *) inputSamples);
        ymm_in_2S = simde_mm256_loadu_si256((simde__m256i *)(inputSamples + 2 * inputStride));
        
        ymm1 = simde_mm256_sad_epu8(ymm_in, ymm_zero);
        ymm2 = simde_mm256_sad_epu8(ymm_in_2S, ymm_zero);

        ymm_sum1 = simde_mm256_add_epi16(ymm1, ymm2);

        inputSamples += 4 * inputStride;
        ymm_in_second    = simde_mm256_loadu_si256((simde__m256i *)inputSamples);
        ymm_in_2S_second = simde_mm256_loadu_si256((simde__m256i *)(inputSamples + 2* inputStride));

        ymm3 = simde_mm256_sad_epu8(ymm_in_second, ymm_zero);
        ymm4 = simde_mm256_sad_epu8(ymm_in_2S_second, ymm_zero);

        ymm_sum2 = simde_mm256_add_epi16(ymm3, ymm4);

        ymm_FinalSum = simde_mm256_add_epi16(ymm_sum1, ymm_sum2);

        ymm_shift = simde_mm256_set_epi64x (3,3,3,3 );
        ymm_FinalSum = simde_mm256_sllv_epi64(ymm_FinalSum,ymm_shift);

        simde_mm256_storeu_si256((simde__m256i *)(meanOf8x8Blocks), ymm_FinalSum);

        /*******************************Squared Mean******************************/
        
        ymm_inputlo = simde_mm256_unpacklo_epi8(ymm_in, ymm_zero);
        ymm_inputhi = simde_mm256_unpackhi_epi8(ymm_in, ymm_zero);
	    
        ymm_blockMeanSquaredlow = simde_mm256_madd_epi16(ymm_inputlo, ymm_inputlo);
        ymm_blockMeanSquaredHi  = simde_mm256_madd_epi16(ymm_inputhi, ymm_inputhi);

        ymm_inputlo = simde_mm256_unpacklo_epi8(ymm_in_2S, ymm_zero);
	    ymm_inputhi = simde_mm256_unpackhi_epi8(ymm_in_2S, ymm_zero);
        
        ymm_blockMeanSquaredlow = simde_mm256_add_epi32(ymm_blockMeanSquaredlow, simde_mm256_madd_epi16(ymm_inputlo, ymm_inputlo));
        ymm_blockMeanSquaredHi  = simde_mm256_add_epi32(ymm_blockMeanSquaredHi, simde_mm256_madd_epi16(ymm_inputhi, ymm_inputhi));

        ymm_inputlo = simde_mm256_unpacklo_epi8(ymm_in_second, ymm_zero);
	    ymm_inputhi = simde_mm256_unpackhi_epi8(ymm_in_second, ymm_zero);
        
        ymm_blockMeanSquaredlow = simde_mm256_add_epi32(ymm_blockMeanSquaredlow, simde_mm256_madd_epi16(ymm_inputlo, ymm_inputlo));
        ymm_blockMeanSquaredHi  = simde_mm256_add_epi32(ymm_blockMeanSquaredHi, simde_mm256_madd_epi16(ymm_inputhi, ymm_inputhi));

        ymm_inputlo = simde_mm256_unpacklo_epi8(ymm_in_2S_second, ymm_zero);
        ymm_inputhi = simde_mm256_unpackhi_epi8(ymm_in_2S_second, ymm_zero);
	    
        ymm_blockMeanSquaredlow = simde_mm256_add_epi32(ymm_blockMeanSquaredlow, simde_mm256_madd_epi16(ymm_inputlo, ymm_inputlo));
        ymm_blockMeanSquaredHi  = simde_mm256_add_epi32(ymm_blockMeanSquaredHi, simde_mm256_madd_epi16(ymm_inputhi, ymm_inputhi));

        ymm_blockMeanSquaredlow = simde_mm256_add_epi32(ymm_blockMeanSquaredlow, simde_mm256_srli_si256(ymm_blockMeanSquaredlow, 8));
	    ymm_blockMeanSquaredHi  = simde_mm256_add_epi32(ymm_blockMeanSquaredHi, simde_mm256_srli_si256(ymm_blockMeanSquaredHi, 8));

        ymm_blockMeanSquaredlow = simde_mm256_add_epi32(ymm_blockMeanSquaredlow, simde_mm256_srli_si256(ymm_blockMeanSquaredlow, 4));
        ymm_blockMeanSquaredHi  = simde_mm256_add_epi32(ymm_blockMeanSquaredHi, simde_mm256_srli_si256(ymm_blockMeanSquaredHi, 4));

        ymm_permute8            = simde_mm256_set_epi32(0,0,0,0,0,0,4,0);
        ymm_blockMeanSquaredlow =  simde_mm256_permutevar8x32_epi32(ymm_blockMeanSquaredlow,ymm_permute8/*8*/);
        ymm_blockMeanSquaredHi  =  simde_mm256_permutevar8x32_epi32(ymm_blockMeanSquaredHi,ymm_permute8);
        
        ymm_blockMeanSquaredlo = simde_mm256_extracti128_si256(ymm_blockMeanSquaredlow,0); //lower 128
        ymm_blockMeanSquaredhi = simde_mm256_extracti128_si256(ymm_blockMeanSquaredHi,0); //lower 128

        ymm_result   = simde_mm256_unpacklo_epi32(simde_mm256_castsi128_si256(ymm_blockMeanSquaredlo),simde_mm256_castsi128_si256(ymm_blockMeanSquaredhi));
        ymm_resultlo = simde_mm_unpacklo_epi64(simde_mm256_castsi256_si128(ymm_result),xmm_zero);
        ymm_resulthi = simde_mm_unpackhi_epi64(simde_mm256_castsi256_si128(ymm_result),xmm_zero);
        
        
        ymm_result   = simde_mm256_set_m128i(ymm_resulthi,ymm_resultlo);
        
        ymm_permute8 = simde_mm256_set_epi32(7,5,6,4,3,1,2,0);
        ymm_result   =  simde_mm256_permutevar8x32_epi32(ymm_result,ymm_permute8);
        
        ymm_shiftSquared = simde_mm256_set1_epi64x (11 );

        ymm_result = simde_mm256_sllv_epi64(ymm_result,ymm_shiftSquared);
        

        simde_mm256_storeu_si256((simde__m256i *)(meanOfSquared8x8Blocks), ymm_result);


     }
