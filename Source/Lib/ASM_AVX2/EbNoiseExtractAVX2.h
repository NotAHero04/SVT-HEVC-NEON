/*
* Copyright(c) 2018 Intel Corporation
* SPDX - License - Identifier: BSD - 2 - Clause - Patent
*/

#ifndef EbNoiseExtractAVX2_h
#define EbNoiseExtractAVX2_h

#include "../../../simde/simde/x86/avx2.h"
#include "EbDefinitions.h"
#include "EbPictureBufferDesc.h"
#ifdef __cplusplus
extern "C" {
#endif


/*******************************************
* noiseExtractLumaWeak
*  weak filter Luma and store noise.
*******************************************/
void noiseExtractLumaWeak_AVX2_INTRIN(
	EbPictureBufferDesc_t       *inputPicturePtr,	
	EbPictureBufferDesc_t       *denoisedPicturePtr,
	EbPictureBufferDesc_t       *noisePicturePtr,   
	EB_U32                       lcuOriginY,
	EB_U32                       lcuOriginX
	);

void noiseExtractLumaWeakLcu_AVX2_INTRIN(
	EbPictureBufferDesc_t       *inputPicturePtr,
	EbPictureBufferDesc_t       *denoisedPicturePtr,
	EbPictureBufferDesc_t       *noisePicturePtr,
	EB_U32                       lcuOriginY,
	EB_U32                       lcuOriginX
	);

void noiseExtractChromaStrong_AVX2_INTRIN(
	EbPictureBufferDesc_t       *inputPicturePtr,
	EbPictureBufferDesc_t       *denoisedPicturePtr,
	EB_U32                       lcuOriginY,
	EB_U32						 lcuOriginX);

void noiseExtractChromaWeak_AVX2_INTRIN(
	EbPictureBufferDesc_t       *inputPicturePtr,
	EbPictureBufferDesc_t       *denoisedPicturePtr,
	EB_U32                       lcuOriginY,
	EB_U32						 lcuOriginX);

void noiseExtractLumaStrong_AVX2_INTRIN(
	EbPictureBufferDesc_t       *inputPicturePtr,
	EbPictureBufferDesc_t       *denoisedPicturePtr,
	EB_U32                       lcuOriginY,
	EB_U32                       lcuOriginX);

void ChromaStrong_AVX2_INTRIN(
	simde__m256i						top,
	simde__m256i						curr,
	simde__m256i						bottom,
	simde__m256i						currPrev,
	simde__m256i						currNext,
	simde__m256i						topPrev,
	simde__m256i						topNext,
	simde__m256i						bottomPrev,
	simde__m256i						bottomNext,
	EB_U8					   *ptrDenoised);

void lumaWeakFilter_AVX2_INTRIN(
	simde__m256i						top,
	simde__m256i						curr,
	simde__m256i						bottom,
	simde__m256i						currPrev,
	simde__m256i						currNext,
	EB_U8					   *ptrDenoised,
	EB_U8					    *ptrNoise);


void chromaWeakLumaStrongFilter_AVX2_INTRIN(
	simde__m256i						top,
	simde__m256i						curr,
	simde__m256i						bottom,
	simde__m256i						currPrev,
	simde__m256i						currNext,
	simde__m256i						topPrev,
	simde__m256i						topNext,
	simde__m256i						bottomPrev,
	simde__m256i						bottomNext,
	EB_U8					   *ptrDenoised);

#ifdef __cplusplus
}
#endif        
#endif // EbNoiseExtractAVX2_h

