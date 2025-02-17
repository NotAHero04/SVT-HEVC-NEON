/*
* Copyright(c) 2018 Intel Corporation
* SPDX - License - Identifier: BSD - 2 - Clause - Patent
*/

#include "EbPictureOperators_SSE2.h"
#include "../../../simde/simde/x86/sse2.h"
#include "EbDefinitions.h"

static simde__m128i simde_mm_loadh_epi64(simde__m128i x, simde__m128i *p)
{
  return simde_mm_castpd_si128(simde_mm_loadh_pd(simde_mm_castsi128_pd(x), (double *)p));
}

// Note: maximum energy within 1 TU considered to be 2^30-e
// All functions can accumulate up to 4 TUs to stay within 32-bit unsigned range

//-------
void FullDistortionKernel4x4_32bit_BT_SSE2(
	EB_S16  *coeff,
	EB_U32   coeffStride,
	EB_S16  *reconCoeff,
	EB_U32   reconCoeffStride,
	EB_U64   distortionResult[2],
	EB_U32   areaWidth,
	EB_U32   areaHeight)
{
  EB_S32 rowCount;
  simde__m128i sum = simde_mm_setzero_si128();
  simde__m128i sum2 = simde_mm_setzero_si128();
  
  rowCount = 2;
  do
  {
    simde__m128i x0;
    simde__m128i y0;
    simde__m128i z0;
    
    x0 = simde_mm_loadl_epi64((simde__m128i *)coeff); coeff += coeffStride;
    x0 = simde_mm_loadh_epi64(x0, (simde__m128i *)coeff); coeff += coeffStride;
    y0 = simde_mm_loadl_epi64((simde__m128i *)reconCoeff); reconCoeff += reconCoeffStride;
    y0 = simde_mm_loadh_epi64(y0, (simde__m128i *)reconCoeff); reconCoeff += reconCoeffStride;
    
    z0 = simde_mm_madd_epi16(x0, x0);
    
    sum2 = simde_mm_add_epi32(sum2, z0);
    
    x0 = simde_mm_sub_epi16(x0, y0);
    
    x0 = simde_mm_madd_epi16(x0, x0);
    
    sum = simde_mm_add_epi32(sum, x0);
  }
  while (--rowCount);
  
  sum = simde_mm_add_epi32(sum, simde_mm_shuffle_epi32(sum, 0x4e)); // 01001110
  sum2 = simde_mm_add_epi32(sum2, simde_mm_shuffle_epi32(sum2, 0x4e)); // 01001110
  sum = simde_mm_unpacklo_epi32(sum, sum2);
  sum = simde_mm_add_epi32(sum, simde_mm_shuffle_epi32(sum, 0x4e)); // 01001110
  simde_mm_storeu_si128((simde__m128i *)distortionResult, simde_mm_unpacklo_epi32(sum, simde_mm_setzero_si128()));

  (void)areaWidth;
  (void)areaHeight;
}

void FullDistortionKernel8x8_32bit_BT_SSE2(
	EB_S16  *coeff,
	EB_U32   coeffStride,
	EB_S16  *reconCoeff,
	EB_U32   reconCoeffStride,
	EB_U64   distortionResult[2],
	EB_U32   areaWidth,
	EB_U32   areaHeight)
{
  EB_S32 rowCount;
  
  simde__m128i sum = simde_mm_setzero_si128();
  simde__m128i sum2 = simde_mm_setzero_si128();
  
  rowCount = 8;
  do
  {
    simde__m128i x0;
    simde__m128i y0;
    simde__m128i z0;
    
    x0 = simde_mm_loadu_si128((simde__m128i *)(coeff + 0x00));
    y0 = simde_mm_loadu_si128((simde__m128i *)(reconCoeff + 0x00));
    coeff += coeffStride;
    reconCoeff += reconCoeffStride;
    
    z0 = simde_mm_madd_epi16(x0, x0);
    
    sum2 = simde_mm_add_epi32(sum2, z0);
    
    x0 = simde_mm_sub_epi16(x0, y0);
    
    x0 = simde_mm_madd_epi16(x0, x0);
    
    sum = simde_mm_add_epi32(sum, x0);
  }
  while (--rowCount);
  
  sum = simde_mm_add_epi32(sum, simde_mm_shuffle_epi32(sum, 0x4e)); // 01001110
  sum2 = simde_mm_add_epi32(sum2, simde_mm_shuffle_epi32(sum2, 0x4e)); // 01001110
  sum = simde_mm_unpacklo_epi32(sum, sum2);
  sum = simde_mm_add_epi32(sum, simde_mm_shuffle_epi32(sum, 0x4e)); // 01001110
  simde_mm_storeu_si128((simde__m128i *)distortionResult, simde_mm_unpacklo_epi32(sum, simde_mm_setzero_si128()));

  (void)areaWidth;
  (void)areaHeight;
}

void FullDistortionKernel16MxN_32bit_BT_SSE2(
	EB_S16  *coeff,
	EB_U32   coeffStride,
	EB_S16  *reconCoeff,
	EB_U32   reconCoeffStride,
	EB_U64   distortionResult[2],
	EB_U32   areaWidth,
	EB_U32   areaHeight)
{
  EB_S32 rowCount, colCount;
  simde__m128i sum = simde_mm_setzero_si128();
  simde__m128i sum2 = simde_mm_setzero_si128();
  
  colCount = areaWidth;
  do
  {
    EB_S16 *coeffTemp = coeff;
    EB_S16 *reconCoeffTemp = reconCoeff;
    
    rowCount = areaHeight;
    do
    {
      simde__m128i x0, x1;
      simde__m128i y0, y1;
      simde__m128i z0, z1;
      
      x0 = simde_mm_loadu_si128((simde__m128i *)(coeffTemp + 0x00));
      x1 = simde_mm_loadu_si128((simde__m128i *)(coeffTemp + 0x08));
      y0 = simde_mm_loadu_si128((simde__m128i *)(reconCoeffTemp + 0x00));
      y1 = simde_mm_loadu_si128((simde__m128i *)(reconCoeffTemp + 0x08));
      coeffTemp += coeffStride;
      reconCoeffTemp += reconCoeffStride;
      
      z0 = simde_mm_madd_epi16(x0, x0);
      z1 = simde_mm_madd_epi16(x1, x1);
      
      sum2 = simde_mm_add_epi32(sum2, z0);
      sum2 = simde_mm_add_epi32(sum2, z1);
      
      x0 = simde_mm_sub_epi16(x0, y0);
      x1 = simde_mm_sub_epi16(x1, y1);
      
      x0 = simde_mm_madd_epi16(x0, x0);
      x1 = simde_mm_madd_epi16(x1, x1);
      
      sum = simde_mm_add_epi32(sum, x0);
      sum = simde_mm_add_epi32(sum, x1);
    }
    while (--rowCount);
    
    coeff += 16;
    reconCoeff += 16;
    colCount -= 16;
  }
  while (colCount > 0);
  
  sum = simde_mm_add_epi32(sum, simde_mm_shuffle_epi32(sum, 0x4e)); // 01001110
  sum2 = simde_mm_add_epi32(sum2, simde_mm_shuffle_epi32(sum2, 0x4e)); // 01001110
  sum = simde_mm_unpacklo_epi32(sum, sum2);
  sum = simde_mm_add_epi32(sum, simde_mm_shuffle_epi32(sum, 0x4e)); // 01001110
  simde_mm_storeu_si128((simde__m128i *)distortionResult, simde_mm_unpacklo_epi32(sum, simde_mm_setzero_si128()));
}


void FullDistortionKernelIntra4x4_32bit_BT_SSE2(
	EB_S16  *coeff,
	EB_U32   coeffStride,
	EB_S16  *reconCoeff,
	EB_U32   reconCoeffStride,
	EB_U64   distortionResult[2],
	EB_U32   areaWidth,
	EB_U32   areaHeight)
{
  EB_S32 rowCount;
 
  simde__m128i sum = simde_mm_setzero_si128();
  
  rowCount = 2;
  do
  {
    simde__m128i x0;
    simde__m128i y0;
    
    x0 = simde_mm_loadl_epi64((simde__m128i *)coeff); coeff += coeffStride;
    x0 = simde_mm_loadh_epi64(x0, (simde__m128i *)coeff); coeff += coeffStride;
    y0 = simde_mm_loadl_epi64((simde__m128i *)reconCoeff); reconCoeff += reconCoeffStride;
    y0 = simde_mm_loadh_epi64(y0, (simde__m128i *)reconCoeff); reconCoeff += reconCoeffStride;
    
    x0 = simde_mm_sub_epi16(x0, y0);
    
    x0 = simde_mm_madd_epi16(x0, x0);
    
    sum = simde_mm_add_epi32(sum, x0);
  }
  while (--rowCount);
  
  sum = simde_mm_add_epi32(sum, simde_mm_shuffle_epi32(sum, 0x4e)); // 01001110
  sum = simde_mm_unpacklo_epi32(sum, sum);
  sum = simde_mm_add_epi32(sum, simde_mm_shuffle_epi32(sum, 0x4e)); // 01001110
  simde_mm_storeu_si128((simde__m128i *)distortionResult, simde_mm_unpacklo_epi32(sum, simde_mm_setzero_si128()));

  (void)areaWidth;
  (void)areaHeight;
}

void FullDistortionKernelIntra8x8_32bit_BT_SSE2(
	EB_S16  *coeff,
	EB_U32   coeffStride,
	EB_S16  *reconCoeff,
	EB_U32   reconCoeffStride,
	EB_U64   distortionResult[2],
	EB_U32   areaWidth,
	EB_U32   areaHeight)
{
  EB_S32 rowCount;
 
  simde__m128i sum = simde_mm_setzero_si128();
  
  rowCount = 8;
  do
  {
    simde__m128i x0;
    simde__m128i y0;
    
    x0 = simde_mm_loadu_si128((simde__m128i *)(coeff + 0x00));
    y0 = simde_mm_loadu_si128((simde__m128i *)(reconCoeff + 0x00));
    coeff += coeffStride;
    reconCoeff += reconCoeffStride;
    
    x0 = simde_mm_sub_epi16(x0, y0);
    
    x0 = simde_mm_madd_epi16(x0, x0);
    
    sum = simde_mm_add_epi32(sum, x0);
  }
  while (--rowCount);
  
  sum = simde_mm_add_epi32(sum, simde_mm_shuffle_epi32(sum, 0x4e)); // 01001110
  sum = simde_mm_unpacklo_epi32(sum, sum);
  sum = simde_mm_add_epi32(sum, simde_mm_shuffle_epi32(sum, 0x4e)); // 01001110
  simde_mm_storeu_si128((simde__m128i *)distortionResult, simde_mm_unpacklo_epi32(sum, simde_mm_setzero_si128()));

  (void)areaWidth;
  (void)areaHeight;
}

void FullDistortionKernelIntra16MxN_32bit_BT_SSE2(
	EB_S16  *coeff,
	EB_U32   coeffStride,
	EB_S16  *reconCoeff,
	EB_U32   reconCoeffStride,
	EB_U64   distortionResult[2],
	EB_U32   areaWidth,
	EB_U32   areaHeight)
{
  EB_S32 rowCount, colCount;
  simde__m128i sum = simde_mm_setzero_si128();
  
  colCount = areaWidth;
  do
  {
    EB_S16 *coeffTemp = coeff;
    EB_S16 *reconCoeffTemp = reconCoeff;
    
    rowCount = areaHeight;
    do
    {
      simde__m128i x0, x1;
      simde__m128i y0, y1;
      
      x0 = simde_mm_loadu_si128((simde__m128i *)(coeffTemp + 0x00));
      x1 = simde_mm_loadu_si128((simde__m128i *)(coeffTemp + 0x08));
      y0 = simde_mm_loadu_si128((simde__m128i *)(reconCoeffTemp + 0x00));
      y1 = simde_mm_loadu_si128((simde__m128i *)(reconCoeffTemp + 0x08));
      coeffTemp += coeffStride;
      reconCoeffTemp += reconCoeffStride;
      
      x0 = simde_mm_sub_epi16(x0, y0);
      x1 = simde_mm_sub_epi16(x1, y1);
      
      x0 = simde_mm_madd_epi16(x0, x0);
      x1 = simde_mm_madd_epi16(x1, x1);
      
      sum = simde_mm_add_epi32(sum, x0);
      sum = simde_mm_add_epi32(sum, x1);
    }
    while (--rowCount);
    
    coeff += 16;
    reconCoeff += 16;
    colCount -= 16;
  }
  while (colCount > 0);
  
  sum = simde_mm_add_epi32(sum, simde_mm_shuffle_epi32(sum, 0x4e)); // 01001110
  sum = simde_mm_unpacklo_epi32(sum, sum);
  sum = simde_mm_add_epi32(sum, simde_mm_shuffle_epi32(sum, 0x4e)); // 01001110
  simde_mm_storeu_si128((simde__m128i *)distortionResult, simde_mm_unpacklo_epi32(sum, simde_mm_setzero_si128()));
}



void FullDistortionKernelCbfZero4x4_32bit_BT_SSE2(
	EB_S16  *coeff,
	EB_U32   coeffStride,
	EB_S16  *reconCoeff,
	EB_U32   reconCoeffStride,
	EB_U64   distortionResult[2],
	EB_U32   areaWidth,
	EB_U32   areaHeight)
{
  EB_S32 rowCount;
  simde__m128i sum2 = simde_mm_setzero_si128();
  
  rowCount = 2;
  do
  {
    simde__m128i x0;
    simde__m128i z0;
    
    x0 = simde_mm_loadl_epi64((simde__m128i *)coeff); coeff += coeffStride;
    x0 = simde_mm_loadh_epi64(x0, (simde__m128i *)coeff); coeff += coeffStride;
    
    z0 = simde_mm_madd_epi16(x0, x0);
    
    sum2 = simde_mm_add_epi32(sum2, z0);
  }
  while (--rowCount);
  
  sum2 = simde_mm_add_epi32(sum2, simde_mm_shuffle_epi32(sum2, 0x4e)); // 01001110
  sum2 = simde_mm_unpacklo_epi32(sum2, sum2);
  sum2 = simde_mm_add_epi32(sum2, simde_mm_shuffle_epi32(sum2, 0x4e)); // 01001110
  simde_mm_storeu_si128((simde__m128i *)distortionResult, simde_mm_unpacklo_epi32(sum2, simde_mm_setzero_si128()));

  (void)areaWidth;
  (void)areaHeight;
  (void)reconCoeff;
  (void)reconCoeffStride;
}

void FullDistortionKernelCbfZero8x8_32bit_BT_SSE2(
	EB_S16  *coeff,
	EB_U32   coeffStride,
	EB_S16  *reconCoeff,
	EB_U32   reconCoeffStride,
	EB_U64   distortionResult[2],
	EB_U32   areaWidth,
	EB_U32   areaHeight)
{
  EB_S32 rowCount;
  simde__m128i sum2 = simde_mm_setzero_si128();
  
  rowCount = 8;
  do
  {
    simde__m128i x0;
    simde__m128i z0;
    
    x0 = simde_mm_loadu_si128((simde__m128i *)(coeff + 0x00));
    coeff += coeffStride;
    
    z0 = simde_mm_madd_epi16(x0, x0);
    
    sum2 = simde_mm_add_epi32(sum2, z0);
  }
  while (--rowCount);
  
  sum2 = simde_mm_add_epi32(sum2, simde_mm_shuffle_epi32(sum2, 0x4e)); // 01001110
  sum2 = simde_mm_unpacklo_epi32(sum2, sum2);
  sum2 = simde_mm_add_epi32(sum2, simde_mm_shuffle_epi32(sum2, 0x4e)); // 01001110
  simde_mm_storeu_si128((simde__m128i *)distortionResult, simde_mm_unpacklo_epi32(sum2, simde_mm_setzero_si128()));

  (void)areaWidth;
  (void)areaHeight;
  (void)reconCoeff;
  (void)reconCoeffStride;
}

void FullDistortionKernelCbfZero16MxN_32bit_BT_SSE2(
	EB_S16  *coeff,
	EB_U32   coeffStride,
	EB_S16  *reconCoeff,
	EB_U32   reconCoeffStride,
	EB_U64   distortionResult[2],
	EB_U32   areaWidth,
	EB_U32   areaHeight)
{
  EB_S32 rowCount, colCount;
  simde__m128i sum2 = simde_mm_setzero_si128();

  colCount = areaWidth;
  do
  {
    EB_S16 *coeffTemp = coeff;
    
    rowCount = areaHeight;
    do
    {
      simde__m128i x0, x1;
      simde__m128i z0, z1;
      
      x0 = simde_mm_loadu_si128((simde__m128i *)(coeffTemp + 0x00));
      x1 = simde_mm_loadu_si128((simde__m128i *)(coeffTemp + 0x08));
      coeffTemp += coeffStride;
      
      z0 = simde_mm_madd_epi16(x0, x0);
      z1 = simde_mm_madd_epi16(x1, x1);
      
      sum2 = simde_mm_add_epi32(sum2, z0);
      sum2 = simde_mm_add_epi32(sum2, z1);
    }
    while (--rowCount);
    
    coeff += 16;
    reconCoeff += 16;
    colCount -= 16;
  }
  while (colCount > 0);
  
  sum2 = simde_mm_add_epi32(sum2, simde_mm_shuffle_epi32(sum2, 0x4e)); // 01001110
  sum2 = simde_mm_unpacklo_epi32(sum2, sum2);
  sum2 = simde_mm_add_epi32(sum2, simde_mm_shuffle_epi32(sum2, 0x4e)); // 01001110
  simde_mm_storeu_si128((simde__m128i *)distortionResult, simde_mm_unpacklo_epi32(sum2, simde_mm_setzero_si128()));
  (void)reconCoeffStride;
  
}

/*******************************************************************************
                         PictureCopyKernel_INTRIN
*******************************************************************************/
void PictureCopyKernel4x4_SSE_INTRIN(
	EB_BYTE                  src,
	EB_U32                   srcStride,
	EB_BYTE                  dst,
	EB_U32                   dstStride,
	EB_U32                   areaWidth,
	EB_U32                   areaHeight)
{		
    *(EB_U32 *)dst = *(EB_U32 *)src;	
    *(EB_U32 *)(dst + dstStride) = *(EB_U32 *)(src + srcStride);	
    *(EB_U32 *)(dst + (dstStride << 1)) = *(EB_U32 *)(src + (srcStride << 1));	
    *(EB_U32 *)(dst + (dstStride * 3)) = *(EB_U32 *)(src + (srcStride * 3));	
		
	(void)areaWidth;
	(void)areaHeight;

	return;
}

void PictureCopyKernel8x8_SSE2_INTRIN(
	EB_BYTE                  src,
	EB_U32                   srcStride,
	EB_BYTE                  dst,
	EB_U32                   dstStride,
	EB_U32                   areaWidth,
	EB_U32                   areaHeight)
{		
	simde_mm_storel_epi64((simde__m128i*)dst, simde_mm_cvtsi64_si128(*(EB_U64 *)src));
    simde_mm_storel_epi64((simde__m128i*)(dst + srcStride), simde_mm_cvtsi64_si128(*(EB_U64 *)(src + srcStride)));
    simde_mm_storel_epi64((simde__m128i*)(dst + (srcStride << 1)), simde_mm_cvtsi64_si128(*(EB_U64 *)(src + (srcStride << 1))));
    simde_mm_storel_epi64((simde__m128i*)(dst + 3*srcStride), simde_mm_cvtsi64_si128(*(EB_U64 *)(src + 3*srcStride)));

    src += (srcStride << 2);
    dst += (dstStride << 2);

    simde_mm_storel_epi64((simde__m128i*)dst, simde_mm_cvtsi64_si128(*(EB_U64 *)src));
    simde_mm_storel_epi64((simde__m128i*)(dst + srcStride), simde_mm_cvtsi64_si128(*(EB_U64 *)(src + srcStride)));
    simde_mm_storel_epi64((simde__m128i*)(dst + (srcStride << 1)), simde_mm_cvtsi64_si128(*(EB_U64 *)(src + (srcStride << 1))));
    simde_mm_storel_epi64((simde__m128i*)(dst + 3*srcStride), simde_mm_cvtsi64_si128(*(EB_U64 *)(src + 3*srcStride)));
	
	(void)areaWidth;
	(void)areaHeight;

	return;
}

void PictureCopyKernel16x16_SSE2_INTRIN(
	EB_BYTE                  src,
	EB_U32                   srcStride,
	EB_BYTE                  dst,
	EB_U32                   dstStride,
	EB_U32                   areaWidth,
	EB_U32                   areaHeight)
{
	simde_mm_storeu_si128((simde__m128i*)dst, simde_mm_loadu_si128((simde__m128i*)src));
    simde_mm_storeu_si128((simde__m128i*)(dst + dstStride), simde_mm_loadu_si128((simde__m128i*)(src + srcStride)));
    simde_mm_storeu_si128((simde__m128i*)(dst + (dstStride << 1)), simde_mm_loadu_si128((simde__m128i*)(src + (srcStride << 1))));
    simde_mm_storeu_si128((simde__m128i*)(dst + (dstStride * 3)), simde_mm_loadu_si128((simde__m128i*)(src + (srcStride * 3))));

	src += (srcStride << 2);
    dst += (dstStride << 2);

   	simde_mm_storeu_si128((simde__m128i*)dst, simde_mm_loadu_si128((simde__m128i*)src));
    simde_mm_storeu_si128((simde__m128i*)(dst + dstStride), simde_mm_loadu_si128((simde__m128i*)(src + srcStride)));
    simde_mm_storeu_si128((simde__m128i*)(dst + (dstStride << 1)), simde_mm_loadu_si128((simde__m128i*)(src + (srcStride << 1))));
    simde_mm_storeu_si128((simde__m128i*)(dst + (dstStride * 3)), simde_mm_loadu_si128((simde__m128i*)(src + (srcStride * 3))));

   	src += (srcStride << 2);
    dst += (dstStride << 2);

    simde_mm_storeu_si128((simde__m128i*)dst, simde_mm_loadu_si128((simde__m128i*)src));
    simde_mm_storeu_si128((simde__m128i*)(dst + dstStride), simde_mm_loadu_si128((simde__m128i*)(src + srcStride)));
    simde_mm_storeu_si128((simde__m128i*)(dst + (dstStride << 1)), simde_mm_loadu_si128((simde__m128i*)(src + (srcStride << 1))));
    simde_mm_storeu_si128((simde__m128i*)(dst + (dstStride * 3)), simde_mm_loadu_si128((simde__m128i*)(src + (srcStride * 3))));

	src += (srcStride << 2);
    dst += (dstStride << 2);

   	simde_mm_storeu_si128((simde__m128i*)dst, simde_mm_loadu_si128((simde__m128i*)src));
    simde_mm_storeu_si128((simde__m128i*)(dst + dstStride), simde_mm_loadu_si128((simde__m128i*)(src + srcStride)));
    simde_mm_storeu_si128((simde__m128i*)(dst + (dstStride << 1)), simde_mm_loadu_si128((simde__m128i*)(src + (srcStride << 1))));
    simde_mm_storeu_si128((simde__m128i*)(dst + (dstStride * 3)), simde_mm_loadu_si128((simde__m128i*)(src + (srcStride * 3))));

   	src += (srcStride << 2);
    dst += (dstStride << 2);

	(void)areaWidth;
	(void)areaHeight;

	return;
}


void PictureCopyKernel32x32_SSE2_INTRIN(
	EB_BYTE                  src,
	EB_U32                   srcStride,
	EB_BYTE                  dst,
	EB_U32                   dstStride,
	EB_U32                   areaWidth,
	EB_U32                   areaHeight)
{
	EB_U32 y;
	
	for (y = 0; y < 4; ++y){
		
		simde_mm_storeu_si128((simde__m128i*)dst, simde_mm_loadu_si128((simde__m128i*)src));
		simde_mm_storeu_si128((simde__m128i*)(dst + 16), simde_mm_loadu_si128((simde__m128i*)(src + 16)));
        simde_mm_storeu_si128((simde__m128i*)(dst + dstStride), simde_mm_loadu_si128((simde__m128i*)(src + srcStride)));
		simde_mm_storeu_si128((simde__m128i*)(dst + dstStride + 16), simde_mm_loadu_si128((simde__m128i*)(src + srcStride + 16)));
        simde_mm_storeu_si128((simde__m128i*)(dst + (dstStride << 1)), simde_mm_loadu_si128((simde__m128i*)(src + (srcStride << 1))));
		simde_mm_storeu_si128((simde__m128i*)(dst + (dstStride << 1) + 16), simde_mm_loadu_si128((simde__m128i*)(src + (srcStride << 1) + 16)));
        simde_mm_storeu_si128((simde__m128i*)(dst + 3 * dstStride), simde_mm_loadu_si128((simde__m128i*)(src + 3 * srcStride)));
		simde_mm_storeu_si128((simde__m128i*)(dst + 3 * dstStride + 16), simde_mm_loadu_si128((simde__m128i*)(src + 3 * srcStride + 16)));

        src += (srcStride << 2);
        dst += (dstStride << 2);

        simde_mm_storeu_si128((simde__m128i*)dst, simde_mm_loadu_si128((simde__m128i*)src));
		simde_mm_storeu_si128((simde__m128i*)(dst + 16), simde_mm_loadu_si128((simde__m128i*)(src + 16)));
        simde_mm_storeu_si128((simde__m128i*)(dst + dstStride), simde_mm_loadu_si128((simde__m128i*)(src + srcStride)));
		simde_mm_storeu_si128((simde__m128i*)(dst + dstStride + 16), simde_mm_loadu_si128((simde__m128i*)(src + srcStride + 16)));
        simde_mm_storeu_si128((simde__m128i*)(dst + (dstStride << 1)), simde_mm_loadu_si128((simde__m128i*)(src + (srcStride << 1))));
		simde_mm_storeu_si128((simde__m128i*)(dst + (dstStride << 1) + 16), simde_mm_loadu_si128((simde__m128i*)(src + (srcStride << 1) + 16)));
        simde_mm_storeu_si128((simde__m128i*)(dst + 3 * dstStride), simde_mm_loadu_si128((simde__m128i*)(src + 3 * srcStride)));
		simde_mm_storeu_si128((simde__m128i*)(dst + 3 * dstStride + 16), simde_mm_loadu_si128((simde__m128i*)(src + 3 * srcStride + 16)));

        src += (srcStride << 2);
        dst += (dstStride << 2);
	}
	(void)areaWidth;
	(void)areaHeight;

	return;
}

void PictureCopyKernel64x64_SSE2_INTRIN(
	EB_BYTE                  src,
	EB_U32                   srcStride,
	EB_BYTE                  dst,
	EB_U32                   dstStride,
	EB_U32                   areaWidth,
	EB_U32                   areaHeight)
{
	EB_U32 y;

	for (y = 0; y < 8; ++y){
		
		simde_mm_storeu_si128((simde__m128i*)dst, simde_mm_loadu_si128((simde__m128i*)src));
		simde_mm_storeu_si128((simde__m128i*)(dst + 16), simde_mm_loadu_si128((simde__m128i*)(src + 16)));
		simde_mm_storeu_si128((simde__m128i*)(dst + 32), simde_mm_loadu_si128((simde__m128i*)(src + 32)));
		simde_mm_storeu_si128((simde__m128i*)(dst + 48), simde_mm_loadu_si128((simde__m128i*)(src + 48)));
        simde_mm_storeu_si128((simde__m128i*)(dst + dstStride), simde_mm_loadu_si128((simde__m128i*)(src + srcStride)));
		simde_mm_storeu_si128((simde__m128i*)(dst + dstStride + 16), simde_mm_loadu_si128((simde__m128i*)(src + srcStride + 16)));
		simde_mm_storeu_si128((simde__m128i*)(dst + dstStride + 32), simde_mm_loadu_si128((simde__m128i*)(src + srcStride + 32)));
		simde_mm_storeu_si128((simde__m128i*)(dst + dstStride + 48), simde_mm_loadu_si128((simde__m128i*)(src + srcStride + 48)));
        simde_mm_storeu_si128((simde__m128i*)(dst + (dstStride << 1)), simde_mm_loadu_si128((simde__m128i*)(src + (srcStride << 1))));
		simde_mm_storeu_si128((simde__m128i*)(dst + (dstStride << 1) + 16), simde_mm_loadu_si128((simde__m128i*)(src + (srcStride << 1) + 16)));
		simde_mm_storeu_si128((simde__m128i*)(dst + (dstStride << 1) + 32), simde_mm_loadu_si128((simde__m128i*)(src + (srcStride << 1) + 32)));
		simde_mm_storeu_si128((simde__m128i*)(dst + (dstStride << 1) + 48), simde_mm_loadu_si128((simde__m128i*)(src + (srcStride << 1) + 48)));
        simde_mm_storeu_si128((simde__m128i*)(dst + 3 * dstStride), simde_mm_loadu_si128((simde__m128i*)(src + 3 * srcStride)));
		simde_mm_storeu_si128((simde__m128i*)(dst + 3 * dstStride + 16), simde_mm_loadu_si128((simde__m128i*)(src + 3 * srcStride + 16)));
		simde_mm_storeu_si128((simde__m128i*)(dst + 3 * dstStride + 32), simde_mm_loadu_si128((simde__m128i*)(src + 3 * srcStride + 32)));
		simde_mm_storeu_si128((simde__m128i*)(dst + 3 * dstStride + 48), simde_mm_loadu_si128((simde__m128i*)(src + 3 * srcStride + 48)));

        src += (srcStride << 2);
        dst += (dstStride << 2);
        
        simde_mm_storeu_si128((simde__m128i*)dst, simde_mm_loadu_si128((simde__m128i*)src));
		simde_mm_storeu_si128((simde__m128i*)(dst + 16), simde_mm_loadu_si128((simde__m128i*)(src + 16)));
		simde_mm_storeu_si128((simde__m128i*)(dst + 32), simde_mm_loadu_si128((simde__m128i*)(src + 32)));
		simde_mm_storeu_si128((simde__m128i*)(dst + 48), simde_mm_loadu_si128((simde__m128i*)(src + 48)));
        simde_mm_storeu_si128((simde__m128i*)(dst + dstStride), simde_mm_loadu_si128((simde__m128i*)(src + srcStride)));
		simde_mm_storeu_si128((simde__m128i*)(dst + dstStride + 16), simde_mm_loadu_si128((simde__m128i*)(src + srcStride + 16)));
		simde_mm_storeu_si128((simde__m128i*)(dst + dstStride + 32), simde_mm_loadu_si128((simde__m128i*)(src + srcStride + 32)));
		simde_mm_storeu_si128((simde__m128i*)(dst + dstStride + 48), simde_mm_loadu_si128((simde__m128i*)(src + srcStride + 48)));
        simde_mm_storeu_si128((simde__m128i*)(dst + (dstStride << 1)), simde_mm_loadu_si128((simde__m128i*)(src + (srcStride << 1))));
		simde_mm_storeu_si128((simde__m128i*)(dst + (dstStride << 1) + 16), simde_mm_loadu_si128((simde__m128i*)(src + (srcStride << 1) + 16)));
		simde_mm_storeu_si128((simde__m128i*)(dst + (dstStride << 1) + 32), simde_mm_loadu_si128((simde__m128i*)(src + (srcStride << 1) + 32)));
		simde_mm_storeu_si128((simde__m128i*)(dst + (dstStride << 1) + 48), simde_mm_loadu_si128((simde__m128i*)(src + (srcStride << 1) + 48)));
        simde_mm_storeu_si128((simde__m128i*)(dst + 3 * dstStride), simde_mm_loadu_si128((simde__m128i*)(src + 3 * srcStride)));
		simde_mm_storeu_si128((simde__m128i*)(dst + 3 * dstStride + 16), simde_mm_loadu_si128((simde__m128i*)(src + 3 * srcStride + 16)));
		simde_mm_storeu_si128((simde__m128i*)(dst + 3 * dstStride + 32), simde_mm_loadu_si128((simde__m128i*)(src + 3 * srcStride + 32)));
		simde_mm_storeu_si128((simde__m128i*)(dst + 3 * dstStride + 48), simde_mm_loadu_si128((simde__m128i*)(src + 3 * srcStride + 48)));

        src += (srcStride << 2);
        dst += (dstStride << 2);
	}
	(void)areaWidth;
	(void)areaHeight;

	return;
}
/*******************************************************************************
                      PictureAdditionKernel_INTRIN
*******************************************************************************/
void PictureAdditionKernel4x4_SSE_INTRIN(
	EB_U8  *predPtr,
	EB_U32  predStride,
	EB_S16 *residualPtr,
	EB_U32  residualStride,
	EB_U8  *reconPtr,
	EB_U32  reconStride,
	EB_U32  width,
	EB_U32  height)
{
	EB_U32 y;
    simde__m128i xmm0, recon_0_3;
	xmm0 = simde_mm_setzero_si128();

	for (y = 0; y < 4; ++y){

		recon_0_3 = simde_mm_packus_epi16(simde_mm_add_epi16(simde_mm_unpacklo_epi8(simde_mm_cvtsi32_si128(*(EB_U32 *)predPtr), xmm0), simde_mm_loadl_epi64((simde__m128i *)residualPtr)), xmm0);

		*(EB_U32 *)reconPtr = simde_mm_cvtsi128_si32(recon_0_3);
		predPtr += predStride;
		residualPtr += residualStride;
		reconPtr += reconStride;
	}
	(void)width;
	(void)height;

	return;
}

void PictureAdditionKernel8x8_SSE2_INTRIN(
	EB_U8  *predPtr,
	EB_U32  predStride,
	EB_S16 *residualPtr,
	EB_U32  residualStride,
	EB_U8  *reconPtr,
	EB_U32  reconStride,
	EB_U32  width,
	EB_U32  height)
{

	simde__m128i recon_0_7, xmm0;
	EB_U32 y;

	xmm0 = simde_mm_setzero_si128();

	for (y = 0; y < 8; ++y){

		recon_0_7 = simde_mm_packus_epi16(simde_mm_add_epi16(simde_mm_unpacklo_epi8(simde_mm_loadl_epi64((simde__m128i *)predPtr), xmm0), simde_mm_loadu_si128((simde__m128i *)residualPtr)), xmm0);

		*(EB_U64 *)reconPtr = simde_mm_cvtsi128_si64(recon_0_7);
		predPtr += predStride;
		residualPtr += residualStride;
		reconPtr += reconStride;
	}
	(void)width;
	(void)height;

	return;
}

void PictureAdditionKernel16x16_SSE2_INTRIN(
	EB_U8  *predPtr,
	EB_U32  predStride,
	EB_S16 *residualPtr,
	EB_U32  residualStride,
	EB_U8  *reconPtr,
	EB_U32  reconStride,
	EB_U32  width,
	EB_U32  height)
{
	simde__m128i xmm0, xmm_clip_U8, pred_0_15, recon_0_7, recon_8_15;
	EB_U32 y;
	
	xmm0 = simde_mm_setzero_si128();

	for (y = 0; y < 16; ++y){

		pred_0_15 = simde_mm_loadu_si128((simde__m128i *)predPtr);
		recon_0_7 = simde_mm_add_epi16(simde_mm_unpacklo_epi8(pred_0_15, xmm0), simde_mm_loadu_si128((simde__m128i *)residualPtr));
		recon_8_15 = simde_mm_add_epi16(simde_mm_unpackhi_epi8(pred_0_15, xmm0), simde_mm_loadu_si128((simde__m128i *)(residualPtr + 8)));
		xmm_clip_U8 = simde_mm_packus_epi16(recon_0_7, recon_8_15);
		
		simde_mm_storeu_si128((simde__m128i*)reconPtr, xmm_clip_U8);

		predPtr += predStride;
		residualPtr += residualStride;
		reconPtr += reconStride;
	}
	(void)width;
	(void)height;

	return;

}
void PictureAdditionKernel32x32_SSE2_INTRIN(
	EB_U8  *predPtr,
	EB_U32  predStride,
	EB_S16 *residualPtr,
	EB_U32  residualStride,
	EB_U8  *reconPtr,
	EB_U32  reconStride,
	EB_U32  width,
	EB_U32  height)
{
	EB_U32 y;
    simde__m128i xmm0, pred_0_15, pred_16_31, recon_0_15_clipped, recon_0_7, recon_8_15, recon_16_23, recon_24_31, recon_16_31_clipped;
	xmm0 = simde_mm_setzero_si128();

	for (y = 0; y < 32; ++y){
		pred_0_15 = simde_mm_loadu_si128((simde__m128i *)predPtr);
        pred_16_31 = simde_mm_loadu_si128((simde__m128i *)(predPtr + 16));

		recon_0_7 = simde_mm_add_epi16(simde_mm_unpacklo_epi8(pred_0_15, xmm0), simde_mm_loadu_si128((simde__m128i *)residualPtr));
		recon_8_15 = simde_mm_add_epi16(simde_mm_unpackhi_epi8(pred_0_15, xmm0), simde_mm_loadu_si128((simde__m128i *)(residualPtr + 8)));
		recon_16_23 = simde_mm_add_epi16(simde_mm_unpacklo_epi8(pred_16_31, xmm0), simde_mm_loadu_si128((simde__m128i *)(residualPtr + 16)));
		recon_24_31 = simde_mm_add_epi16(simde_mm_unpackhi_epi8(pred_16_31, xmm0), simde_mm_loadu_si128((simde__m128i *)(residualPtr + 24)));
        
        recon_0_15_clipped = simde_mm_packus_epi16(recon_0_7, recon_8_15);
		recon_16_31_clipped = simde_mm_packus_epi16(recon_16_23, recon_24_31);

        simde_mm_storeu_si128((simde__m128i*)reconPtr, recon_0_15_clipped);
		simde_mm_storeu_si128((simde__m128i*)(reconPtr + 16), recon_16_31_clipped);

		predPtr += predStride;
		residualPtr += residualStride;
		reconPtr += reconStride;
	}
	(void)width;
	(void)height;

	return;
}

void PictureAdditionKernel64x64_SSE2_INTRIN(
	EB_U8  *predPtr,
	EB_U32  predStride,
	EB_S16 *residualPtr,
	EB_U32  residualStride,
	EB_U8  *reconPtr,
	EB_U32  reconStride,
	EB_U32  width,
	EB_U32  height)
{
	EB_U32 y;

    simde__m128i xmm0, pred_0_15, pred_16_31, pred_32_47, pred_48_63;
    simde__m128i recon_0_15_clipped, recon_16_31_clipped, recon_32_47_clipped, recon_48_63_clipped;
    simde__m128i recon_0_7, recon_8_15, recon_16_23, recon_24_31, recon_32_39, recon_40_47, recon_48_55, recon_56_63;

    xmm0 = simde_mm_setzero_si128();

	for (y = 0; y < 64; ++y){

		pred_0_15 = simde_mm_loadu_si128((simde__m128i *)predPtr);
        pred_16_31 = simde_mm_loadu_si128((simde__m128i *)(predPtr + 16));
        pred_32_47 = simde_mm_loadu_si128((simde__m128i *)(predPtr + 32));
        pred_48_63 = simde_mm_loadu_si128((simde__m128i *)(predPtr + 48));

		recon_0_7 = simde_mm_add_epi16(simde_mm_unpacklo_epi8(pred_0_15, xmm0), simde_mm_loadu_si128((simde__m128i *)residualPtr));
		recon_8_15 = simde_mm_add_epi16(simde_mm_unpackhi_epi8(pred_0_15, xmm0), simde_mm_loadu_si128((simde__m128i *)(residualPtr + 8)));
		recon_16_23 = simde_mm_add_epi16(simde_mm_unpacklo_epi8(pred_16_31, xmm0), simde_mm_loadu_si128((simde__m128i *)(residualPtr + 16)));
		recon_24_31 = simde_mm_add_epi16(simde_mm_unpackhi_epi8(pred_16_31, xmm0), simde_mm_loadu_si128((simde__m128i *)(residualPtr + 24)));
		recon_32_39 = simde_mm_add_epi16(simde_mm_unpacklo_epi8(pred_32_47, xmm0), simde_mm_loadu_si128((simde__m128i *)(residualPtr + 32)));
		recon_40_47 = simde_mm_add_epi16(simde_mm_unpackhi_epi8(pred_32_47, xmm0), simde_mm_loadu_si128((simde__m128i *)(residualPtr + 40)));
		recon_48_55 = simde_mm_add_epi16(simde_mm_unpacklo_epi8(pred_48_63, xmm0), simde_mm_loadu_si128((simde__m128i *)(residualPtr + 48)));
		recon_56_63 = simde_mm_add_epi16(simde_mm_unpackhi_epi8(pred_48_63, xmm0), simde_mm_loadu_si128((simde__m128i *)(residualPtr + 56)));

        recon_0_15_clipped = simde_mm_packus_epi16(recon_0_7, recon_8_15);
        recon_16_31_clipped = simde_mm_packus_epi16(recon_16_23, recon_24_31);
        recon_32_47_clipped = simde_mm_packus_epi16(recon_32_39, recon_40_47);
		recon_48_63_clipped = simde_mm_packus_epi16(recon_48_55, recon_56_63);

        simde_mm_storeu_si128((simde__m128i*)reconPtr, recon_0_15_clipped);
        simde_mm_storeu_si128((simde__m128i*)(reconPtr + 16), recon_16_31_clipped);
        simde_mm_storeu_si128((simde__m128i*)(reconPtr + 32), recon_32_47_clipped);
		simde_mm_storeu_si128((simde__m128i*)(reconPtr + 48), recon_48_63_clipped);

		predPtr += predStride;
		residualPtr += residualStride;
		reconPtr += reconStride;
	}
	(void)width;
	(void)height;

	return;
}

/******************************************************************************************************
ResidualKernel
***********************************************************************************************************/

void ResidualKernel4x4_SSE_INTRIN(
	EB_U8   *input,
	EB_U32   inputStride,
	EB_U8   *pred,
	EB_U32   predStride,
	EB_S16  *residual,
	EB_U32   residualStride,
	EB_U32   areaWidth,
	EB_U32   areaHeight)
{
	simde__m128i residual_0_3, xmm0 = simde_mm_setzero_si128();
	EB_U32 y;

	for (y = 0; y < 4; ++y){

		residual_0_3 = simde_mm_sub_epi16(simde_mm_unpacklo_epi8(simde_mm_cvtsi32_si128(*(EB_U32 *)input), xmm0),
                                     simde_mm_unpacklo_epi8(simde_mm_cvtsi32_si128(*(EB_U32 *)pred), xmm0));

		*(EB_U64 *)residual = simde_mm_cvtsi128_si64(residual_0_3);

		input += inputStride;
		pred += predStride;
		residual += residualStride;
	}
	(void)areaWidth;
	(void)areaHeight;

	return;
}

void ResidualKernel8x8_SSE2_INTRIN(
	EB_U8   *input,
	EB_U32   inputStride,
	EB_U8   *pred,
	EB_U32   predStride,
	EB_S16  *residual,
	EB_U32   residualStride,
	EB_U32   areaWidth,
	EB_U32   areaHeight)
{
	simde__m128i xmm0, residual_0_7;
	EB_U32 y;

	xmm0 = simde_mm_setzero_si128();

	for (y = 0; y < 8; ++y){

		residual_0_7 = simde_mm_sub_epi16(simde_mm_unpacklo_epi8(simde_mm_loadl_epi64((simde__m128i *)input), xmm0), simde_mm_unpacklo_epi8(simde_mm_loadl_epi64((simde__m128i *)pred), xmm0));

		simde_mm_storeu_si128((simde__m128i*)residual, residual_0_7);

		input += inputStride;
		pred += predStride;
		residual += residualStride;
	}
	(void)areaWidth;
	(void)areaHeight;

	return;
}

void ResidualKernel16x16_SSE2_INTRIN(
	EB_U8   *input,
	EB_U32   inputStride,
	EB_U8   *pred,
	EB_U32   predStride,
	EB_S16  *residual,
	EB_U32   residualStride,
	EB_U32   areaWidth,
	EB_U32   areaHeight)
{
    simde__m128i xmm0, residual_0_7, residual_8_15;
	EB_U32 y;
	
	xmm0 = simde_mm_setzero_si128();

	for (y = 0; y < 16; ++y){

		residual_0_7 = simde_mm_sub_epi16( simde_mm_unpacklo_epi8(simde_mm_loadu_si128((simde__m128i *)input), xmm0), simde_mm_unpacklo_epi8(simde_mm_loadu_si128((simde__m128i *)pred), xmm0));
		residual_8_15 = simde_mm_sub_epi16(simde_mm_unpackhi_epi8(simde_mm_loadu_si128((simde__m128i *)input), xmm0), simde_mm_unpackhi_epi8(simde_mm_loadu_si128((simde__m128i *)pred), xmm0));

		simde_mm_storeu_si128((simde__m128i*)residual, residual_0_7);
		simde_mm_storeu_si128((simde__m128i*)(residual + 8), residual_8_15);

		input += inputStride;
		pred += predStride;
		residual += residualStride;
	}
	(void)areaWidth;
	(void)areaHeight;

	return;
}

void ResidualKernel32x32_SSE2_INTRIN(
	EB_U8   *input,
	EB_U32   inputStride,
	EB_U8   *pred,
	EB_U32   predStride,
	EB_S16  *residual,
	EB_U32   residualStride,
	EB_U32   areaWidth,
	EB_U32   areaHeight)
{
	simde__m128i xmm0, residual_0_7, residual_8_15, residual_16_23, residual_24_31;
	EB_U32 y;

	xmm0 = simde_mm_setzero_si128();

	for (y = 0; y < 32; ++y){

		residual_0_7 = simde_mm_sub_epi16(simde_mm_unpacklo_epi8(simde_mm_loadu_si128((simde__m128i *)input), xmm0), simde_mm_unpacklo_epi8(simde_mm_loadu_si128((simde__m128i *)pred), xmm0));
		residual_8_15 = simde_mm_sub_epi16(simde_mm_unpackhi_epi8(simde_mm_loadu_si128((simde__m128i *)input), xmm0), simde_mm_unpackhi_epi8(simde_mm_loadu_si128((simde__m128i *)pred), xmm0));
		residual_16_23 = simde_mm_sub_epi16( simde_mm_unpacklo_epi8(simde_mm_loadu_si128((simde__m128i *)(input + 16)), xmm0), simde_mm_unpacklo_epi8(simde_mm_loadu_si128((simde__m128i *)(pred + 16)), xmm0));
		residual_24_31 = simde_mm_sub_epi16(simde_mm_unpackhi_epi8(simde_mm_loadu_si128((simde__m128i *)(input + 16)), xmm0), simde_mm_unpackhi_epi8(simde_mm_loadu_si128((simde__m128i *)(pred + 16)), xmm0));

        simde_mm_storeu_si128((simde__m128i*)residual, residual_0_7);
		simde_mm_storeu_si128((simde__m128i*)(residual + 8), residual_8_15);
		simde_mm_storeu_si128((simde__m128i*)(residual + 16), residual_16_23);
		simde_mm_storeu_si128((simde__m128i*)(residual + 24), residual_24_31);

		input += inputStride;
		pred += predStride;
		residual += residualStride;
	}
	(void)areaWidth;
	(void)areaHeight;

	return;
}

void ResidualKernel64x64_SSE2_INTRIN(
	EB_U8   *input,
	EB_U32   inputStride,
	EB_U8   *pred,
	EB_U32   predStride,
	EB_S16  *residual,
	EB_U32   residualStride,
	EB_U32   areaWidth,
	EB_U32   areaHeight)
{
	simde__m128i xmm0, residual_0_7, residual_8_15, residual_16_23, residual_24_31, resdiaul_32_39, residual_40_47, residual_48_55, residual_56_63;
	EB_U32 y;

	xmm0 = simde_mm_setzero_si128();

	for (y = 0; y < 64; ++y){

		residual_0_7 = simde_mm_sub_epi16(simde_mm_unpacklo_epi8(simde_mm_loadu_si128((simde__m128i *)input), xmm0), simde_mm_unpacklo_epi8(simde_mm_loadu_si128((simde__m128i *)pred), xmm0));
		residual_8_15 = simde_mm_sub_epi16(simde_mm_unpackhi_epi8(simde_mm_loadu_si128((simde__m128i *)input), xmm0), simde_mm_unpackhi_epi8(simde_mm_loadu_si128((simde__m128i *)pred), xmm0));
		residual_16_23 = simde_mm_sub_epi16(simde_mm_unpacklo_epi8(simde_mm_loadu_si128((simde__m128i *)(input + 16)), xmm0), simde_mm_unpacklo_epi8(simde_mm_loadu_si128((simde__m128i *)(pred + 16)), xmm0));
		residual_24_31 = simde_mm_sub_epi16(simde_mm_unpackhi_epi8(simde_mm_loadu_si128((simde__m128i *)(input + 16)), xmm0), simde_mm_unpackhi_epi8(simde_mm_loadu_si128((simde__m128i *)(pred + 16)), xmm0));
		resdiaul_32_39 = simde_mm_sub_epi16(simde_mm_unpacklo_epi8(simde_mm_loadu_si128((simde__m128i *)(input + 32)), xmm0), simde_mm_unpacklo_epi8(simde_mm_loadu_si128((simde__m128i *)(pred + 32)), xmm0));
		residual_40_47 = simde_mm_sub_epi16(simde_mm_unpackhi_epi8(simde_mm_loadu_si128((simde__m128i *)(input + 32)), xmm0), simde_mm_unpackhi_epi8(simde_mm_loadu_si128((simde__m128i *)(pred + 32)), xmm0));
		residual_48_55 = simde_mm_sub_epi16(simde_mm_unpacklo_epi8(simde_mm_loadu_si128((simde__m128i *)(input + 48)), xmm0), simde_mm_unpacklo_epi8(simde_mm_loadu_si128((simde__m128i *)(pred + 48)), xmm0));
		residual_56_63 = simde_mm_sub_epi16(simde_mm_unpackhi_epi8(simde_mm_loadu_si128((simde__m128i *)(input + 48)), xmm0), simde_mm_unpackhi_epi8(simde_mm_loadu_si128((simde__m128i *)(pred + 48)), xmm0));
        
        simde_mm_storeu_si128((simde__m128i*)residual, residual_0_7);
		simde_mm_storeu_si128((simde__m128i*)(residual + 8), residual_8_15);
		simde_mm_storeu_si128((simde__m128i*)(residual + 16), residual_16_23);
		simde_mm_storeu_si128((simde__m128i*)(residual + 24), residual_24_31);
        simde_mm_storeu_si128((simde__m128i*)(residual + 32), resdiaul_32_39);
		simde_mm_storeu_si128((simde__m128i*)(residual + 40), residual_40_47);
		simde_mm_storeu_si128((simde__m128i*)(residual + 48), residual_48_55);
		simde_mm_storeu_si128((simde__m128i*)(residual + 56), residual_56_63);

		input += inputStride;
		pred += predStride;
		residual += residualStride;
	}
	(void)areaWidth;
	(void)areaHeight;

	return;
}

void ResidualKernelSubSampled4x4_SSE_INTRIN(
	EB_U8   *input,
	EB_U32   inputStride,
	EB_U8   *pred,
	EB_U32   predStride,
	EB_S16  *residual,
	EB_U32   residualStride,
	EB_U32   areaWidth,
	EB_U32   areaHeight,
    EB_U8    lastLine)
{
	simde__m128i residual_0_3, xmm0 = simde_mm_setzero_si128();
	EB_U32 y;
    //hard code subampling dimensions, keep residualStride
    areaHeight>>=1;
    inputStride<<=1;
    predStride<<=1;

	for (y = 0; y < areaHeight; ++y){

		residual_0_3 = simde_mm_sub_epi16(simde_mm_unpacklo_epi8(simde_mm_cvtsi32_si128(*(EB_U32 *)input), xmm0),
                                     simde_mm_unpacklo_epi8(simde_mm_cvtsi32_si128(*(EB_U32 *)pred), xmm0));

		*(EB_U64 *)residual = simde_mm_cvtsi128_si64(residual_0_3);

        residual += residualStride;
        *(EB_U64 *)residual = simde_mm_cvtsi128_si64(residual_0_3);

		input += inputStride;
		pred += predStride;
		residual += residualStride;
	}
	(void)areaWidth;
    //compute the last line

    if(lastLine){
    input -= (inputStride)>>1;
	pred  -= (predStride )>>1;
	residual -= residualStride;	
    residual_0_3 = simde_mm_sub_epi16(simde_mm_unpacklo_epi8(simde_mm_cvtsi32_si128(*(EB_U32 *)input), xmm0),
                                 simde_mm_unpacklo_epi8(simde_mm_cvtsi32_si128(*(EB_U32 *)pred), xmm0));

	*(EB_U64 *)residual = simde_mm_cvtsi128_si64(residual_0_3);
    }

	return;
}

void ResidualKernelSubSampled8x8_SSE2_INTRIN(
	EB_U8   *input,
	EB_U32   inputStride,
	EB_U8   *pred,
	EB_U32   predStride,
	EB_S16  *residual,
	EB_U32   residualStride,
	EB_U32   areaWidth,
	EB_U32   areaHeight,
    EB_U8    lastLine
    
    )
{
	simde__m128i xmm0, residual_0_7;
	EB_U32 y;

	xmm0 = simde_mm_setzero_si128();
    //hard code subampling dimensions, keep residualStride
    areaHeight>>=1;
    inputStride<<=1;
    predStride<<=1;

	for (y = 0; y < areaHeight; ++y){

		residual_0_7 = simde_mm_sub_epi16(simde_mm_unpacklo_epi8(simde_mm_loadl_epi64((simde__m128i *)input), xmm0), simde_mm_unpacklo_epi8(simde_mm_loadl_epi64((simde__m128i *)pred), xmm0));

		simde_mm_storeu_si128((simde__m128i*)residual, residual_0_7);

        residual += residualStride;
        simde_mm_storeu_si128((simde__m128i*)residual, residual_0_7);

		input += inputStride;
		pred += predStride;
		residual += residualStride;
	}
	(void)areaWidth;
    //compute the last line
    if(lastLine){

    input -= (inputStride)>>1;
	pred  -= (predStride )>>1;
	residual -= residualStride;	

    residual_0_7 = simde_mm_sub_epi16(simde_mm_unpacklo_epi8(simde_mm_loadl_epi64((simde__m128i *)input), xmm0), simde_mm_unpacklo_epi8(simde_mm_loadl_epi64((simde__m128i *)pred), xmm0));

	simde_mm_storeu_si128((simde__m128i*)residual, residual_0_7);

    }

	return;
}

void ResidualKernelSubSampled16x16_SSE2_INTRIN(
	EB_U8   *input,
	EB_U32   inputStride,
	EB_U8   *pred,
	EB_U32   predStride,
	EB_S16  *residual,
	EB_U32   residualStride,
	EB_U32   areaWidth,
	EB_U32   areaHeight,
    EB_U8    lastLine
     
    )
{
    simde__m128i xmm0, residual_0_7, residual_8_15;
	EB_U32 y;
	
	xmm0 = simde_mm_setzero_si128();
    //hard code subampling dimensions, keep residualStride
    areaHeight>>=1;
    inputStride<<=1;
    predStride<<=1;

	for (y = 0; y < areaHeight; ++y){

		residual_0_7 = simde_mm_sub_epi16( simde_mm_unpacklo_epi8(simde_mm_loadu_si128((simde__m128i *)input), xmm0), simde_mm_unpacklo_epi8(simde_mm_loadu_si128((simde__m128i *)pred), xmm0));
		residual_8_15 = simde_mm_sub_epi16(simde_mm_unpackhi_epi8(simde_mm_loadu_si128((simde__m128i *)input), xmm0), simde_mm_unpackhi_epi8(simde_mm_loadu_si128((simde__m128i *)pred), xmm0));

		simde_mm_storeu_si128((simde__m128i*)residual, residual_0_7);
		simde_mm_storeu_si128((simde__m128i*)(residual + 8), residual_8_15);

        residual += residualStride;
        simde_mm_storeu_si128((simde__m128i*)residual, residual_0_7);
		simde_mm_storeu_si128((simde__m128i*)(residual + 8), residual_8_15);

		input += inputStride;
		pred += predStride;
		residual += residualStride;
	}
	(void)areaWidth;
    //compute the last line

    if(lastLine){

    input -= (inputStride)>>1;
	pred  -= (predStride )>>1;
	residual -= residualStride;	

    residual_0_7 = simde_mm_sub_epi16( simde_mm_unpacklo_epi8(simde_mm_loadu_si128((simde__m128i *)input), xmm0), simde_mm_unpacklo_epi8(simde_mm_loadu_si128((simde__m128i *)pred), xmm0));
	residual_8_15 = simde_mm_sub_epi16(simde_mm_unpackhi_epi8(simde_mm_loadu_si128((simde__m128i *)input), xmm0), simde_mm_unpackhi_epi8(simde_mm_loadu_si128((simde__m128i *)pred), xmm0));

	simde_mm_storeu_si128((simde__m128i*)residual, residual_0_7);
	simde_mm_storeu_si128((simde__m128i*)(residual + 8), residual_8_15);

    }
	return;
}

void ResidualKernelSubSampled32x32_SSE2_INTRIN(
	EB_U8   *input,
	EB_U32   inputStride,
	EB_U8   *pred,
	EB_U32   predStride,
	EB_S16  *residual,
	EB_U32   residualStride,
	EB_U32   areaWidth,
	EB_U32   areaHeight,
    EB_U8    lastLine)
{
	simde__m128i xmm0, residual_0_7, residual_8_15, residual_16_23, residual_24_31;
	EB_U32 y;

	xmm0 = simde_mm_setzero_si128();
    
    //hard code subampling dimensions, keep residualStride
    areaHeight>>=1;
    inputStride<<=1;
    predStride<<=1;


	for (y = 0; y < areaHeight; ++y){

		residual_0_7 = simde_mm_sub_epi16(simde_mm_unpacklo_epi8(simde_mm_loadu_si128((simde__m128i *)input), xmm0), simde_mm_unpacklo_epi8(simde_mm_loadu_si128((simde__m128i *)pred), xmm0));
		residual_8_15 = simde_mm_sub_epi16(simde_mm_unpackhi_epi8(simde_mm_loadu_si128((simde__m128i *)input), xmm0), simde_mm_unpackhi_epi8(simde_mm_loadu_si128((simde__m128i *)pred), xmm0));
		residual_16_23 = simde_mm_sub_epi16( simde_mm_unpacklo_epi8(simde_mm_loadu_si128((simde__m128i *)(input + 16)), xmm0), simde_mm_unpacklo_epi8(simde_mm_loadu_si128((simde__m128i *)(pred + 16)), xmm0));
		residual_24_31 = simde_mm_sub_epi16(simde_mm_unpackhi_epi8(simde_mm_loadu_si128((simde__m128i *)(input + 16)), xmm0), simde_mm_unpackhi_epi8(simde_mm_loadu_si128((simde__m128i *)(pred + 16)), xmm0));

        simde_mm_storeu_si128((simde__m128i*)residual, residual_0_7);
		simde_mm_storeu_si128((simde__m128i*)(residual + 8), residual_8_15);
		simde_mm_storeu_si128((simde__m128i*)(residual + 16), residual_16_23);
		simde_mm_storeu_si128((simde__m128i*)(residual + 24), residual_24_31);

         residual += residualStride;
        simde_mm_storeu_si128((simde__m128i*)residual, residual_0_7);
		simde_mm_storeu_si128((simde__m128i*)(residual + 8), residual_8_15);
		simde_mm_storeu_si128((simde__m128i*)(residual + 16), residual_16_23);
		simde_mm_storeu_si128((simde__m128i*)(residual + 24), residual_24_31);

		input += inputStride;
		pred += predStride;
		residual += residualStride;
	}
	(void)areaWidth;
        //compute the last line

    if(lastLine){
        input -= (inputStride)>>1;
		pred  -= (predStride )>>1;
		residual -= residualStride;	

        residual_0_7 = simde_mm_sub_epi16(simde_mm_unpacklo_epi8(simde_mm_loadu_si128((simde__m128i *)input), xmm0), simde_mm_unpacklo_epi8(simde_mm_loadu_si128((simde__m128i *)pred), xmm0));
		residual_8_15 = simde_mm_sub_epi16(simde_mm_unpackhi_epi8(simde_mm_loadu_si128((simde__m128i *)input), xmm0), simde_mm_unpackhi_epi8(simde_mm_loadu_si128((simde__m128i *)pred), xmm0));
		residual_16_23 = simde_mm_sub_epi16( simde_mm_unpacklo_epi8(simde_mm_loadu_si128((simde__m128i *)(input + 16)), xmm0), simde_mm_unpacklo_epi8(simde_mm_loadu_si128((simde__m128i *)(pred + 16)), xmm0));
		residual_24_31 = simde_mm_sub_epi16(simde_mm_unpackhi_epi8(simde_mm_loadu_si128((simde__m128i *)(input + 16)), xmm0), simde_mm_unpackhi_epi8(simde_mm_loadu_si128((simde__m128i *)(pred + 16)), xmm0));

        simde_mm_storeu_si128((simde__m128i*)residual, residual_0_7);
		simde_mm_storeu_si128((simde__m128i*)(residual + 8), residual_8_15);
		simde_mm_storeu_si128((simde__m128i*)(residual + 16), residual_16_23);
		simde_mm_storeu_si128((simde__m128i*)(residual + 24), residual_24_31);
    }

	return;
}


void ResidualKernelSubSampled64x64_SSE2_INTRIN(
	EB_U8   *input,
	EB_U32   inputStride,
	EB_U8   *pred,
	EB_U32   predStride,
	EB_S16  *residual,
	EB_U32   residualStride,
	EB_U32   areaWidth,
	EB_U32   areaHeight,
    EB_U8    lastLine)
{
	simde__m128i xmm0, residual_0_7, residual_8_15, residual_16_23, residual_24_31, resdiaul_32_39, residual_40_47, residual_48_55, residual_56_63;
	EB_U32 y;

	xmm0 = simde_mm_setzero_si128();

    //hard code subampling dimensions, keep residualStride
    areaHeight>>=1;
    inputStride<<=1;
    predStride<<=1;

	for (y = 0; y < areaHeight; ++y){

		residual_0_7 = simde_mm_sub_epi16(simde_mm_unpacklo_epi8(simde_mm_loadu_si128((simde__m128i *)input), xmm0), simde_mm_unpacklo_epi8(simde_mm_loadu_si128((simde__m128i *)pred), xmm0));
		residual_8_15 = simde_mm_sub_epi16(simde_mm_unpackhi_epi8(simde_mm_loadu_si128((simde__m128i *)input), xmm0), simde_mm_unpackhi_epi8(simde_mm_loadu_si128((simde__m128i *)pred), xmm0));
		residual_16_23 = simde_mm_sub_epi16(simde_mm_unpacklo_epi8(simde_mm_loadu_si128((simde__m128i *)(input + 16)), xmm0), simde_mm_unpacklo_epi8(simde_mm_loadu_si128((simde__m128i *)(pred + 16)), xmm0));
		residual_24_31 = simde_mm_sub_epi16(simde_mm_unpackhi_epi8(simde_mm_loadu_si128((simde__m128i *)(input + 16)), xmm0), simde_mm_unpackhi_epi8(simde_mm_loadu_si128((simde__m128i *)(pred + 16)), xmm0));
		resdiaul_32_39 = simde_mm_sub_epi16(simde_mm_unpacklo_epi8(simde_mm_loadu_si128((simde__m128i *)(input + 32)), xmm0), simde_mm_unpacklo_epi8(simde_mm_loadu_si128((simde__m128i *)(pred + 32)), xmm0));
		residual_40_47 = simde_mm_sub_epi16(simde_mm_unpackhi_epi8(simde_mm_loadu_si128((simde__m128i *)(input + 32)), xmm0), simde_mm_unpackhi_epi8(simde_mm_loadu_si128((simde__m128i *)(pred + 32)), xmm0));
		residual_48_55 = simde_mm_sub_epi16(simde_mm_unpacklo_epi8(simde_mm_loadu_si128((simde__m128i *)(input + 48)), xmm0), simde_mm_unpacklo_epi8(simde_mm_loadu_si128((simde__m128i *)(pred + 48)), xmm0));
		residual_56_63 = simde_mm_sub_epi16(simde_mm_unpackhi_epi8(simde_mm_loadu_si128((simde__m128i *)(input + 48)), xmm0), simde_mm_unpackhi_epi8(simde_mm_loadu_si128((simde__m128i *)(pred + 48)), xmm0));
        
        simde_mm_storeu_si128((simde__m128i*)residual, residual_0_7);
		simde_mm_storeu_si128((simde__m128i*)(residual + 8), residual_8_15);
		simde_mm_storeu_si128((simde__m128i*)(residual + 16), residual_16_23);
		simde_mm_storeu_si128((simde__m128i*)(residual + 24), residual_24_31);
        simde_mm_storeu_si128((simde__m128i*)(residual + 32), resdiaul_32_39);
		simde_mm_storeu_si128((simde__m128i*)(residual + 40), residual_40_47);
		simde_mm_storeu_si128((simde__m128i*)(residual + 48), residual_48_55);
		simde_mm_storeu_si128((simde__m128i*)(residual + 56), residual_56_63);

//duplicate top field residual to bottom field
         residual += residualStride;
        simde_mm_storeu_si128((simde__m128i*)residual, residual_0_7);
		simde_mm_storeu_si128((simde__m128i*)(residual + 8), residual_8_15);
		simde_mm_storeu_si128((simde__m128i*)(residual + 16), residual_16_23);
		simde_mm_storeu_si128((simde__m128i*)(residual + 24), residual_24_31);
        simde_mm_storeu_si128((simde__m128i*)(residual + 32), resdiaul_32_39);
		simde_mm_storeu_si128((simde__m128i*)(residual + 40), residual_40_47);
		simde_mm_storeu_si128((simde__m128i*)(residual + 48), residual_48_55);
		simde_mm_storeu_si128((simde__m128i*)(residual + 56), residual_56_63);

		input += inputStride;
		pred += predStride;
		residual += residualStride;
	}
	(void)areaWidth;
        //compute the last line

    if(lastLine){
        input -= (inputStride)>>1;
		pred  -= (predStride )>>1;
		residual -= residualStride;

        residual_0_7 = simde_mm_sub_epi16(simde_mm_unpacklo_epi8(simde_mm_loadu_si128((simde__m128i *)input), xmm0), simde_mm_unpacklo_epi8(simde_mm_loadu_si128((simde__m128i *)pred), xmm0));
		residual_8_15 = simde_mm_sub_epi16(simde_mm_unpackhi_epi8(simde_mm_loadu_si128((simde__m128i *)input), xmm0), simde_mm_unpackhi_epi8(simde_mm_loadu_si128((simde__m128i *)pred), xmm0));
		residual_16_23 = simde_mm_sub_epi16(simde_mm_unpacklo_epi8(simde_mm_loadu_si128((simde__m128i *)(input + 16)), xmm0), simde_mm_unpacklo_epi8(simde_mm_loadu_si128((simde__m128i *)(pred + 16)), xmm0));
		residual_24_31 = simde_mm_sub_epi16(simde_mm_unpackhi_epi8(simde_mm_loadu_si128((simde__m128i *)(input + 16)), xmm0), simde_mm_unpackhi_epi8(simde_mm_loadu_si128((simde__m128i *)(pred + 16)), xmm0));
		resdiaul_32_39 = simde_mm_sub_epi16(simde_mm_unpacklo_epi8(simde_mm_loadu_si128((simde__m128i *)(input + 32)), xmm0), simde_mm_unpacklo_epi8(simde_mm_loadu_si128((simde__m128i *)(pred + 32)), xmm0));
		residual_40_47 = simde_mm_sub_epi16(simde_mm_unpackhi_epi8(simde_mm_loadu_si128((simde__m128i *)(input + 32)), xmm0), simde_mm_unpackhi_epi8(simde_mm_loadu_si128((simde__m128i *)(pred + 32)), xmm0));
		residual_48_55 = simde_mm_sub_epi16(simde_mm_unpacklo_epi8(simde_mm_loadu_si128((simde__m128i *)(input + 48)), xmm0), simde_mm_unpacklo_epi8(simde_mm_loadu_si128((simde__m128i *)(pred + 48)), xmm0));
		residual_56_63 = simde_mm_sub_epi16(simde_mm_unpackhi_epi8(simde_mm_loadu_si128((simde__m128i *)(input + 48)), xmm0), simde_mm_unpackhi_epi8(simde_mm_loadu_si128((simde__m128i *)(pred + 48)), xmm0));
        
        simde_mm_storeu_si128((simde__m128i*)residual, residual_0_7);
		simde_mm_storeu_si128((simde__m128i*)(residual + 8), residual_8_15);
		simde_mm_storeu_si128((simde__m128i*)(residual + 16), residual_16_23);
		simde_mm_storeu_si128((simde__m128i*)(residual + 24), residual_24_31);
        simde_mm_storeu_si128((simde__m128i*)(residual + 32), resdiaul_32_39);
		simde_mm_storeu_si128((simde__m128i*)(residual + 40), residual_40_47);
		simde_mm_storeu_si128((simde__m128i*)(residual + 48), residual_48_55);
		simde_mm_storeu_si128((simde__m128i*)(residual + 56), residual_56_63);

    }

	return;
}
/******************************************************************************************************
                                       ResidualKernel16bit_SSE2_INTRIN
******************************************************************************************************/
void ResidualKernel16bit_SSE2_INTRIN(
	EB_U16   *input,
	EB_U32   inputStride,
	EB_U16   *pred,
	EB_U32   predStride,
	EB_S16  *residual,
	EB_U32   residualStride,
	EB_U32   areaWidth,
	EB_U32   areaHeight)
{
	EB_U32 x, y;
    simde__m128i residual0, residual1;
	
	if (areaWidth == 4)
	{
		for (y = 0; y < areaHeight; y += 2){
			
			residual0 = simde_mm_sub_epi16(simde_mm_loadl_epi64((simde__m128i*)input), simde_mm_loadl_epi64((simde__m128i*)pred));
			residual1 = simde_mm_sub_epi16(simde_mm_loadl_epi64((simde__m128i*)(input + inputStride)), simde_mm_loadl_epi64((simde__m128i*)(pred +  predStride)));
			
			simde_mm_storel_epi64((simde__m128i*)residual, residual0);
			simde_mm_storel_epi64((simde__m128i*)(residual + residualStride), residual1);

			input += inputStride << 1;
			pred += predStride << 1;
			residual += residualStride << 1;
		}
	}
	else if (areaWidth == 8){
		for (y = 0; y < areaHeight; y += 2){
				
			residual0 = simde_mm_sub_epi16(simde_mm_loadu_si128((simde__m128i*)input), simde_mm_loadu_si128((simde__m128i*)pred));
			residual1 = simde_mm_sub_epi16(simde_mm_loadu_si128((simde__m128i*)(input + inputStride)), simde_mm_loadu_si128((simde__m128i*)(pred + predStride)));
				
			simde_mm_storeu_si128((simde__m128i*) residual, residual0);
			simde_mm_storeu_si128((simde__m128i*) (residual + residualStride), residual1);

			input += inputStride << 1;
			pred += predStride << 1;
			residual += residualStride << 1;
		}
	}
	else if(areaWidth == 16){

        simde__m128i residual2, residual3;

		for (y = 0; y < areaHeight; y += 2){
					
			residual0 = simde_mm_sub_epi16(simde_mm_loadu_si128((simde__m128i*)input), simde_mm_loadu_si128((simde__m128i*)pred));
			residual1 = simde_mm_sub_epi16(simde_mm_loadu_si128((simde__m128i*)(input + 8)), simde_mm_loadu_si128((simde__m128i*)(pred + 8)));
			residual2 = simde_mm_sub_epi16(simde_mm_loadu_si128((simde__m128i*)(input+inputStride)), simde_mm_loadu_si128((simde__m128i*)(pred+predStride)));
			residual3 = simde_mm_sub_epi16(simde_mm_loadu_si128((simde__m128i*)(input + inputStride+8)), simde_mm_loadu_si128((simde__m128i*)(pred + predStride+8)));
					
			simde_mm_storeu_si128((simde__m128i*)residual, residual0);
			simde_mm_storeu_si128((simde__m128i*)(residual + 8), residual1);
			simde_mm_storeu_si128((simde__m128i*)(residual+residualStride), residual2);
			simde_mm_storeu_si128((simde__m128i*)(residual +residualStride+ 8), residual3);

			input += inputStride << 1;
			pred += predStride << 1;
			residual += residualStride << 1;
		}
	}
	else if(areaWidth == 32){

		for (y = 0; y < areaHeight; y += 2){
            //residual[columnIndex] = ((EB_S16)input[columnIndex]) - ((EB_S16)pred[columnIndex]);
			simde_mm_storeu_si128((simde__m128i*) residual, simde_mm_sub_epi16(simde_mm_loadu_si128((simde__m128i*)input), simde_mm_loadu_si128((simde__m128i*)pred)));
			simde_mm_storeu_si128((simde__m128i*) (residual + 8), simde_mm_sub_epi16(simde_mm_loadu_si128((simde__m128i*)(input + 8)), simde_mm_loadu_si128((simde__m128i*)(pred + 8))));
			simde_mm_storeu_si128((simde__m128i*) (residual + 16), simde_mm_sub_epi16(simde_mm_loadu_si128((simde__m128i*)(input + 16)), simde_mm_loadu_si128((simde__m128i*)(pred + 16))));
			simde_mm_storeu_si128((simde__m128i*) (residual + 24), simde_mm_sub_epi16(simde_mm_loadu_si128((simde__m128i*)(input + 24)), simde_mm_loadu_si128((simde__m128i*)(pred + 24))));

			simde_mm_storeu_si128((simde__m128i*) (residual + residualStride), simde_mm_sub_epi16(simde_mm_loadu_si128((simde__m128i*)(input+inputStride)), simde_mm_loadu_si128((simde__m128i*)(pred+predStride))));
			simde_mm_storeu_si128((simde__m128i*) (residual + residualStride + 8), simde_mm_sub_epi16(simde_mm_loadu_si128((simde__m128i*)(input + inputStride + 8)), simde_mm_loadu_si128((simde__m128i*)(pred+predStride + 8))));
			simde_mm_storeu_si128((simde__m128i*) (residual + residualStride + 16), simde_mm_sub_epi16(simde_mm_loadu_si128((simde__m128i*)(input + inputStride + 16)), simde_mm_loadu_si128((simde__m128i*)(pred + predStride+ 16))));
			simde_mm_storeu_si128((simde__m128i*) (residual + residualStride + 24), simde_mm_sub_epi16(simde_mm_loadu_si128((simde__m128i*)(input + inputStride + 24)), simde_mm_loadu_si128((simde__m128i*)(pred + predStride+ 24))));

			input += inputStride << 1;
			pred += predStride << 1;
			residual += residualStride << 1;
		}
	}			
	else if(areaWidth == 64){ // Branch was not tested because the encoder had max tuSize of 32

		for (y = 0; y < areaHeight; y += 2){

		    //residual[columnIndex] = ((EB_S16)input[columnIndex]) - ((EB_S16)pred[columnIndex]) 8 indices per simde_mm_sub_epi16	
			simde_mm_storeu_si128((simde__m128i*) residual,  simde_mm_sub_epi16(simde_mm_loadu_si128((simde__m128i*)input), simde_mm_loadu_si128((simde__m128i*)pred)));
			simde_mm_storeu_si128((simde__m128i*) (residual + 8), simde_mm_sub_epi16(simde_mm_loadu_si128((simde__m128i*)(input + 8)), simde_mm_loadu_si128((simde__m128i*)(pred + 8))));
			simde_mm_storeu_si128((simde__m128i*) (residual + 16), simde_mm_sub_epi16(simde_mm_loadu_si128((simde__m128i*)(input + 16)), simde_mm_loadu_si128((simde__m128i*)(pred + 16))));
			simde_mm_storeu_si128((simde__m128i*) (residual + 24), simde_mm_sub_epi16(simde_mm_loadu_si128((simde__m128i*)(input + 24)), simde_mm_loadu_si128((simde__m128i*)(pred + 24))));
			simde_mm_storeu_si128((simde__m128i*) (residual + 32), simde_mm_sub_epi16(simde_mm_loadu_si128((simde__m128i*)(input + 32)), simde_mm_loadu_si128((simde__m128i*)(pred + 32))));
			simde_mm_storeu_si128((simde__m128i*) (residual + 40), simde_mm_sub_epi16(simde_mm_loadu_si128((simde__m128i*)(input + 40)), simde_mm_loadu_si128((simde__m128i*)(pred + 40))));
			simde_mm_storeu_si128((simde__m128i*) (residual + 48), simde_mm_sub_epi16(simde_mm_loadu_si128((simde__m128i*)(input + 48)), simde_mm_loadu_si128((simde__m128i*)(pred + 48))));
			simde_mm_storeu_si128((simde__m128i*) (residual + 56), simde_mm_sub_epi16(simde_mm_loadu_si128((simde__m128i*)(input + 56)), simde_mm_loadu_si128((simde__m128i*)(pred + 56))));

			simde_mm_storeu_si128((simde__m128i*) (residual + residualStride), simde_mm_sub_epi16(simde_mm_loadu_si128((simde__m128i*)(input + inputStride)), simde_mm_loadu_si128((simde__m128i*)(pred + predStride))));
			simde_mm_storeu_si128((simde__m128i*) (residual + residualStride + 8), simde_mm_sub_epi16(simde_mm_loadu_si128((simde__m128i*)(input + inputStride + 8)), simde_mm_loadu_si128((simde__m128i*)(pred + predStride + 8))));
			simde_mm_storeu_si128((simde__m128i*) (residual + residualStride + 16), simde_mm_sub_epi16(simde_mm_loadu_si128((simde__m128i*)(input + inputStride + 16)), simde_mm_loadu_si128((simde__m128i*)(pred + predStride + 16))));
			simde_mm_storeu_si128((simde__m128i*) (residual + residualStride + 24), simde_mm_sub_epi16(simde_mm_loadu_si128((simde__m128i*)(input + inputStride + 24)), simde_mm_loadu_si128((simde__m128i*)(pred + predStride + 24))));
			simde_mm_storeu_si128((simde__m128i*) (residual + residualStride + 32), simde_mm_sub_epi16(simde_mm_loadu_si128((simde__m128i*)(input + inputStride + 32)), simde_mm_loadu_si128((simde__m128i*)(pred + predStride + 32))));
			simde_mm_storeu_si128((simde__m128i*) (residual + residualStride + 40), simde_mm_sub_epi16(simde_mm_loadu_si128((simde__m128i*)(input + inputStride + 40)), simde_mm_loadu_si128((simde__m128i*)(pred + predStride + 40))));
			simde_mm_storeu_si128((simde__m128i*) (residual + residualStride + 48), simde_mm_sub_epi16(simde_mm_loadu_si128((simde__m128i*)(input + inputStride + 48)), simde_mm_loadu_si128((simde__m128i*)(pred + predStride + 48))));
			simde_mm_storeu_si128((simde__m128i*) (residual + residualStride + 56), simde_mm_sub_epi16(simde_mm_loadu_si128((simde__m128i*)(input + inputStride + 56)), simde_mm_loadu_si128((simde__m128i*)(pred + predStride + 56))));

			input += inputStride << 1;
			pred += predStride << 1;
			residual += residualStride << 1;
		}
	}
	else {

		EB_U32 inputStrideDiff = 2 * inputStride;
		EB_U32 predStrideDiff = 2 * predStride;
		EB_U32 residualStrideDiff = 2 * residualStride;
		inputStrideDiff -= areaWidth;
		predStrideDiff -= areaWidth;
		residualStrideDiff -= areaWidth;

		if (!(areaWidth & 7)){

			for (x = 0; x < areaHeight; x += 2){
				for (y = 0; y < areaWidth; y += 8){
					
					simde_mm_storeu_si128((simde__m128i*) residual, simde_mm_sub_epi16(simde_mm_loadu_si128((simde__m128i*)input), simde_mm_loadu_si128((simde__m128i*)pred)));
					simde_mm_storeu_si128((simde__m128i*) (residual + residualStride), simde_mm_sub_epi16(simde_mm_loadu_si128((simde__m128i*)(input + inputStride)), simde_mm_loadu_si128((simde__m128i*)(pred + predStride))));

					input += 8;
					pred += 8;
					residual += 8;
				}
				input = input + inputStrideDiff;
				pred = pred + predStrideDiff;
				residual = residual + residualStrideDiff;
			}
		}
		else{
			for (x = 0; x < areaHeight; x += 2){
				for (y = 0; y < areaWidth; y += 4){
				
					simde_mm_storel_epi64((simde__m128i*) residual, simde_mm_sub_epi16(simde_mm_loadu_si128((simde__m128i*)input), simde_mm_loadu_si128((simde__m128i*)pred)));
					simde_mm_storel_epi64((simde__m128i*) (residual + residualStride), simde_mm_sub_epi16(simde_mm_loadu_si128((simde__m128i*)(input + inputStride)), simde_mm_loadu_si128((simde__m128i*)(pred + predStride))));

					input += 4;
					pred += 4;
					residual += 4;
				}
				input = input + inputStrideDiff;
				pred = pred + predStrideDiff;
				residual = residual + residualStrideDiff;
			}
		}
	}
	return;
}

/******************************************************************************************************
                                   PictureAdditionKernel16bit_SSE2_INTRIN
******************************************************************************************************/


void PictureAdditionKernel16bit_SSE2_INTRIN(
	EB_U16  *predPtr,
	EB_U32  predStride,
	EB_S16 *residualPtr,
	EB_U32  residualStride,
	EB_U16  *reconPtr,
	EB_U32  reconStride,
	EB_U32  width,
	EB_U32  height)
{
    simde__m128i xmm_0, xmm_Max10bit;
    
	EB_U32 y, x;

	xmm_0 = simde_mm_setzero_si128();
	xmm_Max10bit = simde_mm_set1_epi16(1023);

	if (width == 4)
	{
        simde__m128i xmm_sum_0_3, xmm_sum_s0_s3, xmm_clip3_0_3, xmm_clip3_s0_s3;
		for (y = 0; y < height; y += 2){

			xmm_sum_0_3 = simde_mm_adds_epi16(simde_mm_loadl_epi64((simde__m128i*)predPtr), simde_mm_loadl_epi64((simde__m128i*)residualPtr));
			xmm_sum_s0_s3 = simde_mm_adds_epi16(simde_mm_loadl_epi64((simde__m128i*)(predPtr + predStride)), simde_mm_loadl_epi64((simde__m128i*)(residualPtr + residualStride)));

			xmm_clip3_0_3 = simde_mm_max_epi16(simde_mm_min_epi16(xmm_sum_0_3, xmm_Max10bit), xmm_0);
			xmm_clip3_s0_s3 = simde_mm_max_epi16(simde_mm_min_epi16(xmm_sum_s0_s3, xmm_Max10bit), xmm_0);

			simde_mm_storel_epi64((simde__m128i*) reconPtr, xmm_clip3_0_3);
			simde_mm_storel_epi64((simde__m128i*) (reconPtr + reconStride), xmm_clip3_s0_s3);

			predPtr += predStride << 1;
			residualPtr += residualStride << 1;
			reconPtr += reconStride << 1;
		}
	}
	else if (width == 8){

        simde__m128i xmm_sum_0_7, xmm_sum_s0_s7, xmm_clip3_0_7, xmm_clip3_s0_s7;

		for (y = 0; y < height; y += 2){

			xmm_sum_0_7 = simde_mm_adds_epi16( simde_mm_loadu_si128((simde__m128i*)predPtr),simde_mm_loadu_si128((simde__m128i*)residualPtr));
			xmm_sum_s0_s7 = simde_mm_adds_epi16(simde_mm_loadu_si128((simde__m128i*)(predPtr + predStride)), simde_mm_loadu_si128((simde__m128i*)(residualPtr + residualStride)));
			
			xmm_clip3_0_7 = simde_mm_max_epi16(simde_mm_min_epi16(xmm_sum_0_7, xmm_Max10bit), xmm_0);
			xmm_clip3_s0_s7 = simde_mm_max_epi16(simde_mm_min_epi16(xmm_sum_s0_s7, xmm_Max10bit), xmm_0);
			
			simde_mm_storeu_si128((simde__m128i*) reconPtr, xmm_clip3_0_7);
			simde_mm_storeu_si128((simde__m128i*) (reconPtr + reconStride), xmm_clip3_s0_s7);

			predPtr += predStride << 1;
			residualPtr += residualStride << 1;
			reconPtr += reconStride << 1;
		}
	}
	else if (width == 16){
        
        simde__m128i sum_0_7, sum_8_15, sum_s0_s7, sum_s8_s15, clip3_0_7, clip3_8_15, clip3_s0_s7, clip3_s8_s15;

		for (y = 0; y < height; y += 2){

			sum_0_7 = simde_mm_adds_epi16(simde_mm_loadu_si128((simde__m128i*)predPtr), simde_mm_loadu_si128((simde__m128i*)residualPtr));
			sum_8_15 = simde_mm_adds_epi16(simde_mm_loadu_si128((simde__m128i*)(predPtr + 8)), simde_mm_loadu_si128((simde__m128i*)(residualPtr + 8)));
			sum_s0_s7 = simde_mm_adds_epi16(simde_mm_loadu_si128((simde__m128i*)(predPtr + predStride)), simde_mm_loadu_si128((simde__m128i*)(residualPtr + residualStride)));
			sum_s8_s15 = simde_mm_adds_epi16(simde_mm_loadu_si128((simde__m128i*)(predPtr + predStride + 8)), simde_mm_loadu_si128((simde__m128i*)(residualPtr + residualStride + 8)));

			clip3_0_7 = simde_mm_max_epi16(simde_mm_min_epi16(sum_0_7, xmm_Max10bit), xmm_0);
			clip3_8_15 = simde_mm_max_epi16(simde_mm_min_epi16(sum_8_15, xmm_Max10bit), xmm_0);
			clip3_s0_s7 = simde_mm_max_epi16(simde_mm_min_epi16(sum_s0_s7, xmm_Max10bit), xmm_0);
			clip3_s8_s15 = simde_mm_max_epi16(simde_mm_min_epi16(sum_s8_s15, xmm_Max10bit), xmm_0);

			simde_mm_storeu_si128((simde__m128i*) reconPtr, clip3_0_7);
			simde_mm_storeu_si128((simde__m128i*) (reconPtr + 8), clip3_8_15);
			simde_mm_storeu_si128((simde__m128i*) (reconPtr + reconStride), clip3_s0_s7);
			simde_mm_storeu_si128((simde__m128i*) (reconPtr + reconStride + 8), clip3_s8_s15);

			predPtr += predStride << 1;
			residualPtr += residualStride << 1;
			reconPtr += reconStride << 1;
		}
	}
	else if (width == 32){
        simde__m128i sum_0_7, sum_8_15, sum_16_23, sum_24_31, sum_s0_s7, sum_s8_s15, sum_s16_s23, sum_s24_s31;
        simde__m128i clip3_0_7, clip3_8_15, clip3_16_23, clip3_24_31, clip3_s0_s7, clip3_s8_s15, clip3_s16_s23, clip3_s24_s31;

		for (y = 0; y < height; y += 2){

			sum_0_7   = simde_mm_adds_epi16(simde_mm_loadu_si128((simde__m128i*)predPtr), simde_mm_loadu_si128((simde__m128i*)residualPtr));
			sum_8_15  = simde_mm_adds_epi16(simde_mm_loadu_si128((simde__m128i*)(predPtr + 8)), simde_mm_loadu_si128((simde__m128i*)(residualPtr + 8)));
			sum_16_23 = simde_mm_adds_epi16(simde_mm_loadu_si128((simde__m128i*)(predPtr + 16)), simde_mm_loadu_si128((simde__m128i*)(residualPtr + 16)));
			sum_24_31 = simde_mm_adds_epi16(simde_mm_loadu_si128((simde__m128i*)(predPtr + 24)), simde_mm_loadu_si128((simde__m128i*)(residualPtr + 24)));
			
			sum_s0_s7   = simde_mm_adds_epi16(simde_mm_loadu_si128((simde__m128i*)(predPtr + predStride)), simde_mm_loadu_si128((simde__m128i*)(residualPtr + residualStride)));
			sum_s8_s15  = simde_mm_adds_epi16(simde_mm_loadu_si128((simde__m128i*)(predPtr + predStride + 8)), simde_mm_loadu_si128((simde__m128i*)(residualPtr + residualStride + 8)));
			sum_s16_s23 = simde_mm_adds_epi16(simde_mm_loadu_si128((simde__m128i*)(predPtr + predStride + 16)), simde_mm_loadu_si128((simde__m128i*)(residualPtr + residualStride + 16)));
			sum_s24_s31 = simde_mm_adds_epi16(simde_mm_loadu_si128((simde__m128i*)(predPtr + predStride + 24)), simde_mm_loadu_si128((simde__m128i*)(residualPtr + residualStride + 24))); 

			clip3_0_7   = simde_mm_max_epi16(simde_mm_min_epi16(sum_0_7, xmm_Max10bit), xmm_0);
			clip3_8_15  = simde_mm_max_epi16(simde_mm_min_epi16(sum_8_15 , xmm_Max10bit), xmm_0);
			clip3_16_23 = simde_mm_max_epi16(simde_mm_min_epi16(sum_16_23, xmm_Max10bit), xmm_0);
			clip3_24_31 = simde_mm_max_epi16(simde_mm_min_epi16(sum_24_31, xmm_Max10bit), xmm_0);
			
			clip3_s0_s7   = simde_mm_max_epi16(simde_mm_min_epi16(sum_s0_s7, xmm_Max10bit), xmm_0);
			clip3_s8_s15  = simde_mm_max_epi16(simde_mm_min_epi16(sum_s8_s15, xmm_Max10bit), xmm_0);
			clip3_s16_s23 = simde_mm_max_epi16(simde_mm_min_epi16(sum_s16_s23, xmm_Max10bit), xmm_0);
			clip3_s24_s31 = simde_mm_max_epi16(simde_mm_min_epi16(sum_s24_s31, xmm_Max10bit), xmm_0);

			simde_mm_storeu_si128((simde__m128i*) reconPtr,        clip3_0_7);
			simde_mm_storeu_si128((simde__m128i*) (reconPtr + 8),  clip3_8_15);
			simde_mm_storeu_si128((simde__m128i*) (reconPtr + 16), clip3_16_23);
			simde_mm_storeu_si128((simde__m128i*) (reconPtr + 24), clip3_24_31);
			
			simde_mm_storeu_si128((simde__m128i*) (reconPtr + reconStride),      clip3_s0_s7);
			simde_mm_storeu_si128((simde__m128i*) (reconPtr + reconStride + 8),  clip3_s8_s15);
			simde_mm_storeu_si128((simde__m128i*) (reconPtr + reconStride + 16), clip3_s16_s23);
			simde_mm_storeu_si128((simde__m128i*) (reconPtr + reconStride + 24), clip3_s24_s31);
			
			predPtr += predStride << 1;
			residualPtr += residualStride << 1;
			reconPtr += reconStride << 1;
		}
	}
	else if (width == 64){ // Branch not tested due to Max TU size is 32 at time of development

        simde__m128i sum_0_7, sum_8_15, sum_16_23, sum_24_31, sum_32_39, sum_40_47, sum_48_55, sum_56_63;
        simde__m128i clip3_0_7, clip3_8_15, clip3_16_23, clip3_24_31, clip3_32_39, clip3_40_47, clip3_48_55, clip3_56_63;

		for (y = 0; y < height; ++y ){

			sum_0_7   = simde_mm_adds_epi16(simde_mm_loadu_si128((simde__m128i*)predPtr), simde_mm_loadu_si128((simde__m128i*)residualPtr));
			sum_8_15  = simde_mm_adds_epi16(simde_mm_loadu_si128((simde__m128i*)(predPtr + 8)), simde_mm_loadu_si128((simde__m128i*)(residualPtr + 8)));
			sum_16_23 = simde_mm_adds_epi16(simde_mm_loadu_si128((simde__m128i*)(predPtr + 16)), simde_mm_loadu_si128((simde__m128i*)(residualPtr + 16)));
			sum_24_31 = simde_mm_adds_epi16(simde_mm_loadu_si128((simde__m128i*)(predPtr + 24)), simde_mm_loadu_si128((simde__m128i*)(residualPtr + 24)));
			sum_32_39 = simde_mm_adds_epi16(simde_mm_loadu_si128((simde__m128i*)(predPtr + 32)), simde_mm_loadu_si128((simde__m128i*)(residualPtr + 32)));
			sum_40_47 = simde_mm_adds_epi16(simde_mm_loadu_si128((simde__m128i*)(predPtr + 40)), simde_mm_loadu_si128((simde__m128i*)(residualPtr + 40)));
			sum_48_55 = simde_mm_adds_epi16(simde_mm_loadu_si128((simde__m128i*)(predPtr + 48)), simde_mm_loadu_si128((simde__m128i*)(residualPtr + 48)));
			sum_56_63 = simde_mm_adds_epi16(simde_mm_loadu_si128((simde__m128i*)(predPtr + 56)), simde_mm_loadu_si128((simde__m128i*)(residualPtr + 56)));

			clip3_0_7   = simde_mm_max_epi16(simde_mm_min_epi16(sum_0_7  , xmm_Max10bit), xmm_0);
			clip3_8_15  = simde_mm_max_epi16(simde_mm_min_epi16(sum_8_15 , xmm_Max10bit), xmm_0);
			clip3_16_23 = simde_mm_max_epi16(simde_mm_min_epi16(sum_16_23, xmm_Max10bit), xmm_0);
			clip3_24_31 = simde_mm_max_epi16(simde_mm_min_epi16(sum_24_31, xmm_Max10bit), xmm_0);
			clip3_32_39 = simde_mm_max_epi16(simde_mm_min_epi16(sum_32_39, xmm_Max10bit), xmm_0);
			clip3_40_47 = simde_mm_max_epi16(simde_mm_min_epi16(sum_40_47, xmm_Max10bit), xmm_0);
			clip3_48_55 = simde_mm_max_epi16(simde_mm_min_epi16(sum_48_55, xmm_Max10bit), xmm_0);
			clip3_56_63 = simde_mm_max_epi16(simde_mm_min_epi16(sum_56_63, xmm_Max10bit), xmm_0);

			simde_mm_storeu_si128((simde__m128i*) reconPtr,        clip3_0_7  );
			simde_mm_storeu_si128((simde__m128i*) (reconPtr + 8),  clip3_8_15 );
			simde_mm_storeu_si128((simde__m128i*) (reconPtr + 16), clip3_16_23);
			simde_mm_storeu_si128((simde__m128i*) (reconPtr + 24), clip3_24_31);
			simde_mm_storeu_si128((simde__m128i*) (reconPtr + 32), clip3_32_39);
			simde_mm_storeu_si128((simde__m128i*) (reconPtr + 40), clip3_40_47);
			simde_mm_storeu_si128((simde__m128i*) (reconPtr + 48), clip3_48_55);
			simde_mm_storeu_si128((simde__m128i*) (reconPtr + 56), clip3_56_63);		

			predPtr += predStride ;
			residualPtr += residualStride ;
			reconPtr += reconStride ;
		}
	}
	else
	{
		EB_U32 predStrideDiff = 2 * predStride;
		EB_U32 residualStrideDiff = 2 * residualStride;
		EB_U32 reconStrideDiff = 2 * reconStride;
		predStrideDiff -= width;
		residualStrideDiff -= width;
		reconStrideDiff -= width;

		if (!(width & 7)){

            simde__m128i xmm_sum_0_7, xmm_sum_s0_s7, xmm_clip3_0_7, xmm_clip3_s0_s7;

			for (x = 0; x < height; x += 2){
				for (y = 0; y < width; y += 8){

					xmm_sum_0_7 = simde_mm_adds_epi16(simde_mm_loadu_si128((simde__m128i*)predPtr), simde_mm_loadu_si128((simde__m128i*)residualPtr));
					xmm_sum_s0_s7 = simde_mm_adds_epi16(simde_mm_loadu_si128((simde__m128i*)(predPtr + predStride)), simde_mm_loadu_si128((simde__m128i*)(residualPtr + residualStride )));

					xmm_clip3_0_7 = simde_mm_max_epi16(simde_mm_min_epi16(xmm_sum_0_7, xmm_Max10bit), xmm_0);
					xmm_clip3_s0_s7 = simde_mm_max_epi16(simde_mm_min_epi16(xmm_sum_s0_s7, xmm_Max10bit), xmm_0);

					simde_mm_storeu_si128((simde__m128i*) reconPtr, xmm_clip3_0_7);
					simde_mm_storeu_si128((simde__m128i*) (reconPtr + reconStride), xmm_clip3_s0_s7);

					predPtr += 8;
					residualPtr += 8;
					reconPtr += 8;
				}
				predPtr += predStrideDiff;
				residualPtr +=  residualStrideDiff;
				reconPtr +=  reconStrideDiff;
			}
		}
		else{
            simde__m128i xmm_sum_0_7, xmm_sum_s0_s7,  xmm_clip3_0_3, xmm_clip3_s0_s3;
			for (x = 0; x < height; x += 2){
				for (y = 0; y < width; y += 4){

					xmm_sum_0_7 = simde_mm_adds_epi16(simde_mm_loadu_si128((simde__m128i*)predPtr), simde_mm_loadu_si128((simde__m128i*)residualPtr));
					xmm_sum_s0_s7 = simde_mm_adds_epi16(simde_mm_loadu_si128((simde__m128i*)(predPtr + predStride)), simde_mm_loadu_si128((simde__m128i*)(residualPtr + residualStride)));

					xmm_clip3_0_3 = simde_mm_max_epi16(simde_mm_min_epi16(xmm_sum_0_7, xmm_Max10bit), xmm_0);
					xmm_clip3_s0_s3 = simde_mm_max_epi16(simde_mm_min_epi16(xmm_sum_s0_s7, xmm_Max10bit), xmm_0);

					simde_mm_storel_epi64((simde__m128i*) reconPtr, xmm_clip3_0_3);
					simde_mm_storel_epi64((simde__m128i*) (reconPtr + reconStride), xmm_clip3_s0_s3);

					predPtr += 4;
					residualPtr += 4;
					reconPtr += 4;
				}
				predPtr +=  predStrideDiff;
				residualPtr +=  residualStrideDiff;
				reconPtr += reconStrideDiff;
			}
		}
	}
	return;
}


