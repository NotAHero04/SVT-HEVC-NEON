/*
* Copyright(c) 2018 Intel Corporation
* SPDX - License - Identifier: BSD - 2 - Clause - Patent
*/


#include "EbNoiseExtractAVX2.h"
#include "EbDefinitions.h"
#include "../../../simde/simde/x86/avx2.h"
#include "EbUtility.h"

EB_EXTERN EB_ALIGN(16) const EB_U8 EbHevcFilterType[] = {
	1, 4, 1, 4, 1, 4, 1, 4, 1, 4, 1, 4, 1, 4, 1, 4, 1, 4, 1, 4, 1, 4, 1, 4, 1, 4, 1, 4, 1, 4, 1, 4
};

EB_EXTERN EB_ALIGN(16) const EB_U8 EbHevcWeakChromafilter[2][32] = {
		{ 2, 4, 2, 4, 2, 4, 2, 4, 2, 4, 2, 4, 2, 4, 2, 4, 2, 4, 2, 4, 2, 4, 2, 4, 2, 4, 2, 4, 2, 4, 2, 4 },
		{ 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2 },
};

inline void lumaWeakFilter_AVX2_INTRIN(

	simde__m256i						top,
	simde__m256i						curr,
	simde__m256i						bottom,
	simde__m256i						currPrev,
	simde__m256i						currNext,
	EB_U8					   *ptrDenoised,
	EB_U8					    *ptrNoise
	)
{
	simde__m256i  topFirstHalf, bottomFirstHalf,
		filterFirstHalf, filterSecondHalf,
		currNextFirstHalf, currNextSecondHalf,
		weights, currLeftMidFirstHalfWeight,
		currLeftMidFirstHalflo, currLeftMidFirstHalfhi, currPrevPermutation, currPermutation, currNextPermutation,
		topPermutation, bottomPermutation;

	currPrevPermutation = simde_mm256_permute4x64_epi64(currPrev, 216);
	currPermutation = simde_mm256_permute4x64_epi64(curr, 216);
	currLeftMidFirstHalflo = simde_mm256_unpacklo_epi8(currPrevPermutation, currPermutation);
	weights = simde_mm256_loadu_si256((simde__m256i*)EbHevcFilterType);
	currLeftMidFirstHalfWeight = simde_mm256_maddubs_epi16(currLeftMidFirstHalflo, weights);
	currNextPermutation = simde_mm256_permute4x64_epi64(currNext, 88);
	currNextFirstHalf = simde_mm256_unpacklo_epi8(currNextPermutation, simde_mm256_setzero_si256());
	currLeftMidFirstHalflo = simde_mm256_add_epi16(currNextFirstHalf, currLeftMidFirstHalfWeight);

	currLeftMidFirstHalfhi = simde_mm256_unpackhi_epi8(currPrevPermutation, currPermutation);
	currLeftMidFirstHalfWeight = simde_mm256_maddubs_epi16(currLeftMidFirstHalfhi, weights);
	currNextPermutation = simde_mm256_permute4x64_epi64(currNext, 216);
	currNextSecondHalf = simde_mm256_unpackhi_epi8(currNextPermutation, simde_mm256_setzero_si256());
	currLeftMidFirstHalfhi = simde_mm256_add_epi16(currNextSecondHalf, currLeftMidFirstHalfWeight);


	topPermutation = simde_mm256_permute4x64_epi64(top, 216);
	topFirstHalf = simde_mm256_unpacklo_epi8(topPermutation, simde_mm256_setzero_si256());
	bottomPermutation = simde_mm256_permute4x64_epi64(bottom, 216);
	bottomFirstHalf = simde_mm256_unpacklo_epi8(bottomPermutation, simde_mm256_setzero_si256());
	filterFirstHalf = simde_mm256_adds_epi16(simde_mm256_adds_epi16(bottomFirstHalf, topFirstHalf), currLeftMidFirstHalflo);
	filterFirstHalf = simde_mm256_srli_epi16(filterFirstHalf, 3);


	topFirstHalf = simde_mm256_unpackhi_epi8(topPermutation, simde_mm256_setzero_si256());
	bottomFirstHalf = simde_mm256_unpackhi_epi8(bottomPermutation, simde_mm256_setzero_si256());
	filterSecondHalf = simde_mm256_adds_epi16(simde_mm256_adds_epi16(bottomFirstHalf, topFirstHalf), currLeftMidFirstHalfhi);
	filterSecondHalf = simde_mm256_srli_epi16(filterSecondHalf, 3);

	filterFirstHalf = simde_mm256_permute4x64_epi64(simde_mm256_packus_epi16(filterFirstHalf, filterSecondHalf), 216);
	simde_mm256_storeu_si256((simde__m256i *)(ptrDenoised ), filterFirstHalf);

	simde_mm256_storeu_si256((simde__m256i *)(ptrNoise), simde_mm256_subs_epu8(curr, filterFirstHalf));

}
inline void chromaWeakLumaStrongFilter_AVX2_INTRIN(

	simde__m256i						top,
	simde__m256i						curr,
	simde__m256i						bottom,
	simde__m256i						currPrev,
	simde__m256i						currNext,
	simde__m256i						topPrev,
	simde__m256i						topNext,
	simde__m256i						bottomPrev,
	simde__m256i						bottomNext,
	EB_U8					   *ptrDenoised
	)
{
	simde__m256i filterFirstHalf, filterSecondHalf,
		currNextFirstHalf, currNextSecondHalf,
		weights, currLeftMidFirstHalfWeight,
		currLeftMidFirstHalflo, currLeftMidFirstHalfhi, currPrevPermutation, currPermutation, currNextPermutation,
		topPermutation, bottomPermutation,
		topPrevPermutation, topLeftMidFirstHalflo, topLeftMidFirstHalfWeight, topNextFirstHalf,
		topNextPermutation, topLeftMidFirstHalfhi, topNextSecondHalf,
		bottomPrevPermutation, bottomLeftMidFirstHalflo, bottomLeftMidFirstHalfWeight, bottomNextPermutation,
		bottomNextFirstHalf, bottomLeftMidFirstHalfhi, bottomNextSecondHalf;


	//  Curr
	currPrevPermutation = simde_mm256_permute4x64_epi64(currPrev, 216);
	currPermutation = simde_mm256_permute4x64_epi64(curr, 216);
	currLeftMidFirstHalflo = simde_mm256_unpacklo_epi8(currPrevPermutation, currPermutation);
	weights = simde_mm256_loadu_si256((simde__m256i*)EbHevcWeakChromafilter[0]);
	currLeftMidFirstHalfWeight = simde_mm256_maddubs_epi16(currLeftMidFirstHalflo, weights);
	currNextPermutation = simde_mm256_permute4x64_epi64(currNext, 88);
	currNextFirstHalf = simde_mm256_unpacklo_epi8(currNextPermutation, simde_mm256_setzero_si256());
	currNextFirstHalf = simde_mm256_slli_epi16(currNextFirstHalf, 1);
	currLeftMidFirstHalflo = simde_mm256_add_epi16(currNextFirstHalf, currLeftMidFirstHalfWeight);

	currLeftMidFirstHalfhi = simde_mm256_unpackhi_epi8(currPrevPermutation, currPermutation);
	currLeftMidFirstHalfWeight = simde_mm256_maddubs_epi16(currLeftMidFirstHalfhi, weights);
	currNextPermutation = simde_mm256_permute4x64_epi64(currNext, 216);
	currNextSecondHalf = simde_mm256_unpackhi_epi8(currNextPermutation, simde_mm256_setzero_si256());
	currNextSecondHalf = simde_mm256_slli_epi16(currNextSecondHalf, 1);
	currLeftMidFirstHalfhi = simde_mm256_add_epi16(currNextSecondHalf, currLeftMidFirstHalfWeight);

	// Top
	topPrevPermutation = simde_mm256_permute4x64_epi64(topPrev, 216);
	topPermutation = simde_mm256_permute4x64_epi64(top, 216);
	topLeftMidFirstHalflo = simde_mm256_unpacklo_epi8(topPrevPermutation, topPermutation);
	weights = simde_mm256_loadu_si256((simde__m256i*)EbHevcWeakChromafilter[1]);
	topLeftMidFirstHalfWeight = simde_mm256_maddubs_epi16(topLeftMidFirstHalflo, weights);
	topNextPermutation = simde_mm256_permute4x64_epi64(topNext, 88);
	topNextFirstHalf = simde_mm256_unpacklo_epi8(topNextPermutation, simde_mm256_setzero_si256());
	topLeftMidFirstHalflo = simde_mm256_add_epi16(topNextFirstHalf, topLeftMidFirstHalfWeight);

	topLeftMidFirstHalfhi = simde_mm256_unpackhi_epi8(topPrevPermutation, topPermutation);
	topLeftMidFirstHalfWeight = simde_mm256_maddubs_epi16(topLeftMidFirstHalfhi, weights);
	topNextPermutation = simde_mm256_permute4x64_epi64(topNext, 216);
	topNextSecondHalf = simde_mm256_unpackhi_epi8(topNextPermutation, simde_mm256_setzero_si256());
	topLeftMidFirstHalfhi = simde_mm256_add_epi16(topNextSecondHalf, topLeftMidFirstHalfWeight);


	// Bottom
	bottomPrevPermutation = simde_mm256_permute4x64_epi64(bottomPrev, 216);
	bottomPermutation = simde_mm256_permute4x64_epi64(bottom, 216);
	bottomLeftMidFirstHalflo = simde_mm256_unpacklo_epi8(bottomPrevPermutation, bottomPermutation);
	weights = simde_mm256_loadu_si256((simde__m256i*)EbHevcWeakChromafilter[1]);
	bottomLeftMidFirstHalfWeight = simde_mm256_maddubs_epi16(bottomLeftMidFirstHalflo, weights);
	bottomNextPermutation = simde_mm256_permute4x64_epi64(bottomNext, 88);
	bottomNextFirstHalf = simde_mm256_unpacklo_epi8(bottomNextPermutation, simde_mm256_setzero_si256());
	bottomLeftMidFirstHalflo = simde_mm256_add_epi16(bottomNextFirstHalf, bottomLeftMidFirstHalfWeight);

	bottomLeftMidFirstHalfhi = simde_mm256_unpackhi_epi8(bottomPrevPermutation, bottomPermutation);
	bottomLeftMidFirstHalfWeight = simde_mm256_maddubs_epi16(bottomLeftMidFirstHalfhi, weights);
	bottomNextPermutation = simde_mm256_permute4x64_epi64(bottomNext, 216);
	bottomNextSecondHalf = simde_mm256_unpackhi_epi8(bottomNextPermutation, simde_mm256_setzero_si256());
	bottomLeftMidFirstHalfhi = simde_mm256_add_epi16(bottomNextSecondHalf, bottomLeftMidFirstHalfWeight);


	filterFirstHalf = simde_mm256_adds_epi16(simde_mm256_adds_epi16(bottomLeftMidFirstHalflo, topLeftMidFirstHalflo), currLeftMidFirstHalflo);
	filterFirstHalf = simde_mm256_srli_epi16(filterFirstHalf, 4);
	filterSecondHalf = simde_mm256_adds_epi16(simde_mm256_adds_epi16(bottomLeftMidFirstHalfhi, topLeftMidFirstHalfhi), currLeftMidFirstHalfhi);
	filterSecondHalf = simde_mm256_srli_epi16(filterSecondHalf, 4);


	filterFirstHalf = simde_mm256_permute4x64_epi64(simde_mm256_packus_epi16(filterFirstHalf, filterSecondHalf), 216);
	simde_mm256_storeu_si256((simde__m256i *)(ptrDenoised), filterFirstHalf);


}

inline void ChromaStrong_AVX2_INTRIN(

	simde__m256i						top,
	simde__m256i						curr,
	simde__m256i						bottom,
	simde__m256i						currPrev,
	simde__m256i						currNext,
	simde__m256i						topPrev,
	simde__m256i						topNext,
	simde__m256i						bottomPrev,
	simde__m256i						bottomNext,
	EB_U8					   *ptrDenoised
	)
{
	simde__m256i   currLeftMidFirstHalflo, currLeftMidFirstHalfhi, currPrevPermutation, currPermutation, currNextPermutation,
		topPermutation, topPrevPermutation, topLeftMidFirstHalflo, topNextPermutation, topLeftMidFirstHalfhi,
		bottomPermutation, bottomPrevPermutation, bottomLeftMidFirstHalflo, bottomNextPermutation, bottomLeftMidFirstHalfhi;


	currPrevPermutation = simde_mm256_permute4x64_epi64(currPrev, 216);
	currPermutation = simde_mm256_permute4x64_epi64(curr, 216);
	currNextPermutation = simde_mm256_permute4x64_epi64(currNext, 216);

	currLeftMidFirstHalflo = simde_mm256_add_epi16(simde_mm256_unpacklo_epi8(currPermutation, simde_mm256_setzero_si256()),
		simde_mm256_unpacklo_epi8(currPrevPermutation, simde_mm256_setzero_si256()));
	currLeftMidFirstHalflo = simde_mm256_add_epi16(simde_mm256_unpacklo_epi8(currNextPermutation, simde_mm256_setzero_si256()), currLeftMidFirstHalflo);

	currLeftMidFirstHalfhi = simde_mm256_add_epi16(simde_mm256_unpackhi_epi8(currPermutation, simde_mm256_setzero_si256()),
		simde_mm256_unpackhi_epi8(currPrevPermutation, simde_mm256_setzero_si256()));
	currLeftMidFirstHalfhi = simde_mm256_add_epi16(simde_mm256_unpackhi_epi8(currNextPermutation, simde_mm256_setzero_si256()), currLeftMidFirstHalfhi);


	topPrevPermutation = simde_mm256_permute4x64_epi64(topPrev, 216);
	topPermutation = simde_mm256_permute4x64_epi64(top, 216);
	topNextPermutation = simde_mm256_permute4x64_epi64(topNext, 216);


	topLeftMidFirstHalflo = simde_mm256_add_epi16(simde_mm256_unpacklo_epi8(topPermutation, simde_mm256_setzero_si256()),
		simde_mm256_unpacklo_epi8(topPrevPermutation, simde_mm256_setzero_si256()));
	topLeftMidFirstHalflo = simde_mm256_add_epi16(simde_mm256_unpacklo_epi8(topNextPermutation, simde_mm256_setzero_si256()), topLeftMidFirstHalflo);


	topLeftMidFirstHalfhi = simde_mm256_add_epi16(simde_mm256_unpackhi_epi8(topPermutation, simde_mm256_setzero_si256()),
		simde_mm256_unpackhi_epi8(topPrevPermutation, simde_mm256_setzero_si256()));
	topLeftMidFirstHalfhi = simde_mm256_add_epi16(simde_mm256_unpackhi_epi8(topNextPermutation, simde_mm256_setzero_si256()), topLeftMidFirstHalfhi);



	bottomPrevPermutation = simde_mm256_permute4x64_epi64(bottomPrev, 216);
	bottomPermutation = simde_mm256_permute4x64_epi64(bottom, 216);
	bottomNextPermutation = simde_mm256_permute4x64_epi64(bottomNext, 216);

	bottomLeftMidFirstHalflo = simde_mm256_add_epi16(simde_mm256_unpacklo_epi8(bottomPermutation, simde_mm256_setzero_si256()),
		simde_mm256_unpacklo_epi8(bottomPrevPermutation, simde_mm256_setzero_si256()));
	bottomLeftMidFirstHalflo = simde_mm256_add_epi16(simde_mm256_unpacklo_epi8(bottomNextPermutation, simde_mm256_setzero_si256()), bottomLeftMidFirstHalflo);


	bottomLeftMidFirstHalfhi = simde_mm256_add_epi16(simde_mm256_unpackhi_epi8(bottomPermutation, simde_mm256_setzero_si256()),
		simde_mm256_unpackhi_epi8(bottomPrevPermutation, simde_mm256_setzero_si256()));
	bottomLeftMidFirstHalfhi = simde_mm256_add_epi16(simde_mm256_unpackhi_epi8(bottomNextPermutation, simde_mm256_setzero_si256()), bottomLeftMidFirstHalfhi);


	currLeftMidFirstHalflo = simde_mm256_add_epi16(simde_mm256_add_epi16(currLeftMidFirstHalflo, topLeftMidFirstHalflo), bottomLeftMidFirstHalflo);
	currLeftMidFirstHalfhi = simde_mm256_add_epi16(simde_mm256_add_epi16(currLeftMidFirstHalfhi, topLeftMidFirstHalfhi), bottomLeftMidFirstHalfhi);

	topLeftMidFirstHalflo = simde_mm256_unpacklo_epi16(currLeftMidFirstHalflo, simde_mm256_setzero_si256());
	topLeftMidFirstHalflo = simde_mm256_mullo_epi32(topLeftMidFirstHalflo, simde_mm256_set1_epi32(7282));
	topLeftMidFirstHalflo = simde_mm256_srli_epi32(topLeftMidFirstHalflo, 16);
	bottomLeftMidFirstHalflo = simde_mm256_unpackhi_epi16(currLeftMidFirstHalflo, simde_mm256_setzero_si256());
	bottomLeftMidFirstHalflo = simde_mm256_mullo_epi32(bottomLeftMidFirstHalflo, simde_mm256_set1_epi32(7282));
	bottomLeftMidFirstHalflo = simde_mm256_srli_epi32(bottomLeftMidFirstHalflo, 16);
	currLeftMidFirstHalflo = simde_mm256_packus_epi32(topLeftMidFirstHalflo, bottomLeftMidFirstHalflo);

	currLeftMidFirstHalflo = simde_mm256_insertf128_si256(simde_mm256_setzero_si256(), simde_mm_packus_epi16(simde_mm256_extracti128_si256(currLeftMidFirstHalflo, 0), simde_mm256_extracti128_si256(currLeftMidFirstHalflo, 1)), 0);


	topLeftMidFirstHalfhi = simde_mm256_unpacklo_epi16(currLeftMidFirstHalfhi, simde_mm256_setzero_si256());
	topLeftMidFirstHalfhi = simde_mm256_mullo_epi32(topLeftMidFirstHalfhi, simde_mm256_set1_epi32(7282));
	topLeftMidFirstHalfhi = simde_mm256_srli_epi32(topLeftMidFirstHalfhi, 16);

	bottomLeftMidFirstHalfhi = simde_mm256_unpackhi_epi16(currLeftMidFirstHalfhi, simde_mm256_setzero_si256());
	bottomLeftMidFirstHalfhi = simde_mm256_mullo_epi32(bottomLeftMidFirstHalfhi, simde_mm256_set1_epi32(7282));
	bottomLeftMidFirstHalfhi = simde_mm256_srli_epi32(bottomLeftMidFirstHalfhi, 16);
	currLeftMidFirstHalfhi = simde_mm256_packus_epi32(topLeftMidFirstHalfhi, bottomLeftMidFirstHalfhi);

	currLeftMidFirstHalflo = simde_mm256_insertf128_si256(currLeftMidFirstHalflo, simde_mm_packus_epi16(simde_mm256_extracti128_si256(currLeftMidFirstHalfhi, 0), simde_mm256_extracti128_si256(currLeftMidFirstHalfhi, 1)), 1);
	simde_mm256_storeu_si256((simde__m256i *)(ptrDenoised), currLeftMidFirstHalflo);


}
/*******************************************
* noiseExtractLumaWeak
*  weak filter Luma and store noise.
*******************************************/
void noiseExtractLumaWeak_AVX2_INTRIN(
	EbPictureBufferDesc_t       *inputPicturePtr,
	EbPictureBufferDesc_t       *denoisedPicturePtr,
	EbPictureBufferDesc_t       *noisePicturePtr,
	EB_U32                       lcuOriginY,
	EB_U32						 lcuOriginX
	)
{
	EB_U32  ii, jj, kk;
	EB_U32  picHeight, lcuHeight;
	EB_U32  picWidth;
	EB_U32  inputOriginIndex;
	EB_U32  inputOriginIndexPad;
	EB_U32  noiseOriginIndex;

	EB_U8 *ptrIn;
	EB_U32 strideIn;
	EB_U8 *ptrDenoised, *ptrDenoisedInterm;

	EB_U8 *ptrNoise, *ptrNoiseInterm;
	EB_U32 strideOut;

	simde__m256i top, curr, bottom, currPrev, currNext,
		secondtop, secondcurr, secondbottom, secondcurrPrev, secondcurrNext;
	(void)lcuOriginX;

	//Luma
	{
		picHeight = inputPicturePtr->height;
		picWidth = inputPicturePtr->width;
		lcuHeight = MIN(MAX_LCU_SIZE, picHeight - lcuOriginY);
		lcuHeight = ((lcuOriginY + MAX_LCU_SIZE >= picHeight) || (lcuOriginY == 0)) ? lcuHeight - 1 : lcuHeight;
		strideIn = inputPicturePtr->strideY;
		inputOriginIndex = inputPicturePtr->originX + (inputPicturePtr->originY + lcuOriginY) * inputPicturePtr->strideY;
		ptrIn = &(inputPicturePtr->bufferY[inputOriginIndex]);

		inputOriginIndexPad = denoisedPicturePtr->originX + (denoisedPicturePtr->originY + lcuOriginY) * denoisedPicturePtr->strideY;
		strideOut = denoisedPicturePtr->strideY;
		ptrDenoised = &(denoisedPicturePtr->bufferY[inputOriginIndexPad]);
		ptrDenoisedInterm = ptrDenoised;

		noiseOriginIndex = noisePicturePtr->originX + noisePicturePtr->originY * noisePicturePtr->strideY;
		ptrNoise = &(noisePicturePtr->bufferY[noiseOriginIndex]);
		ptrNoiseInterm = ptrNoise;

		////Luma
		//a = (p[1] +
		//	p[0 + stride] + 4 * p[1 + stride] + p[2 + stride] +
		//	p[1 + 2 * stride]) / 8;

		top = curr =  secondtop = secondcurr = simde_mm256_setzero_si256();

		for (kk = 0; kk + MAX_LCU_SIZE <= picWidth; kk += MAX_LCU_SIZE)
		{
			for (jj = 0; jj < lcuHeight; jj++)
			{
				if (lcuOriginY == 0)
				{
					if (jj == 0)
					{
						top = simde_mm256_loadu_si256((simde__m256i*)(ptrIn + kk + jj*strideIn));
						secondtop = simde_mm256_loadu_si256((simde__m256i*)(ptrIn + kk + 32 + jj*strideIn));
						curr = simde_mm256_loadu_si256((simde__m256i*)(ptrIn + kk + (1 + jj)*strideIn));
						secondcurr = simde_mm256_loadu_si256((simde__m256i*)((ptrIn + kk + 32) + (1 + jj)*strideIn));
						simde_mm256_storeu_si256((simde__m256i *)(ptrDenoised + kk), top);
						simde_mm256_storeu_si256((simde__m256i *)(ptrDenoised + kk + 32), secondtop);
						simde_mm256_storeu_si256((simde__m256i *)(ptrNoise + kk), simde_mm256_setzero_si256());
						simde_mm256_storeu_si256((simde__m256i *)(ptrNoise + kk + 32), simde_mm256_setzero_si256());
					}
					currPrev = simde_mm256_loadu_si256((simde__m256i*)(ptrIn - 1 + kk + ((1 + jj)*strideIn)));
					currNext = simde_mm256_loadu_si256((simde__m256i*)(ptrIn + 1 + kk + ((1 + jj)*strideIn)));
					secondcurrPrev = simde_mm256_loadu_si256((simde__m256i*)((ptrIn + kk + 32) - 1 + ((1 + jj)*strideIn)));
					secondcurrNext = simde_mm256_loadu_si256((simde__m256i*)((ptrIn + kk + 32) + 1 + ((1 + jj)*strideIn)));
					bottom = simde_mm256_loadu_si256((simde__m256i*)((ptrIn + kk) + (2 + jj)* strideIn));
					secondbottom = simde_mm256_loadu_si256((simde__m256i*)((ptrIn + kk + 32) + (2 + jj)* strideIn));
					ptrDenoisedInterm = ptrDenoised + kk + ((1 + jj)*strideOut);
					ptrNoiseInterm = ptrNoise + kk + ((1 + jj)*strideOut);

				}
				else
				{
					if (jj == 0)
					{
						top = simde_mm256_loadu_si256((simde__m256i*)(ptrIn + kk + jj*strideIn - strideIn));
						secondtop = simde_mm256_loadu_si256((simde__m256i*)(ptrIn + kk + 32 + jj *strideIn - strideIn));
						curr = simde_mm256_loadu_si256((simde__m256i*)(ptrIn + kk + (1 + jj)*strideIn - strideIn));
						secondcurr = simde_mm256_loadu_si256((simde__m256i*)((ptrIn + kk + 32) + (1 + jj)*strideIn - strideIn));
					}
					currPrev = simde_mm256_loadu_si256((simde__m256i*)(ptrIn - 1 + kk + ((1 + jj)*strideIn - strideIn)));
					currNext = simde_mm256_loadu_si256((simde__m256i*)(ptrIn + 1 + kk + ((1 + jj)*strideIn - strideIn)));
					secondcurrPrev = simde_mm256_loadu_si256((simde__m256i*)((ptrIn + kk + 32) - 1 + ((1 + jj)*strideIn - strideIn)));
					secondcurrNext = simde_mm256_loadu_si256((simde__m256i*)((ptrIn + kk + 32) + 1 + ((1 + jj)*strideIn - strideIn)));
					bottom = simde_mm256_loadu_si256((simde__m256i*)((ptrIn + kk) + (2 + jj)* strideIn - strideIn));
					secondbottom = simde_mm256_loadu_si256((simde__m256i*)((ptrIn + kk + 32) + (2 + jj)* strideIn - strideIn));
					ptrDenoisedInterm = ptrDenoised + kk + ((1 + jj)*strideOut - strideOut);
					ptrNoiseInterm = ptrNoise + kk + jj*strideOut;

				}

				lumaWeakFilter_AVX2_INTRIN(
					top,
					curr,
					bottom,
					currPrev,
					currNext,
					ptrDenoisedInterm,
					ptrNoiseInterm);

				lumaWeakFilter_AVX2_INTRIN(
					secondtop,
					secondcurr,
					secondbottom,
					secondcurrPrev,
					secondcurrNext,
					ptrDenoisedInterm + 32,
					ptrNoiseInterm + 32);

				top = curr;
				curr = bottom;
				secondtop = secondcurr;
				secondcurr = secondbottom;

			}


		}

		lcuHeight = MIN(MAX_LCU_SIZE, picHeight - lcuOriginY);

		for (jj = 0; jj < lcuHeight; jj++){
			for (ii = 0; ii < picWidth; ii++){

				if (!((jj<lcuHeight - 1 || lcuOriginY + lcuHeight<picHeight) && ii>0 && ii < picWidth - 1)){

					ptrDenoised[ii + jj*strideOut] = ptrIn[ii + jj*strideIn];
					ptrNoise[ii + jj*strideOut] = 0;
				}

			}
		}

	}

}

void noiseExtractLumaWeakLcu_AVX2_INTRIN(
	EbPictureBufferDesc_t       *inputPicturePtr,
	EbPictureBufferDesc_t       *denoisedPicturePtr,
	EbPictureBufferDesc_t       *noisePicturePtr,
	EB_U32                       lcuOriginY,
	EB_U32						 lcuOriginX
	)
{
	EB_U32  ii, jj;
	EB_U32  picHeight, lcuHeight;
	EB_U32  picWidth, lcuWidth;
	EB_U32  inputOriginIndex;
	EB_U32  inputOriginIndexPad;
	EB_U32  noiseOriginIndex;

	EB_U8 *ptrIn;
	EB_U32 strideIn;
	EB_U8 *ptrDenoised, *ptrDenoisedInterm;

	EB_U8 *ptrNoise, *ptrNoiseInterm;
	EB_U32 strideOut;

	simde__m256i top, curr, bottom, currPrev, currNext,
		secondtop, secondcurr, secondbottom, secondcurrPrev, secondcurrNext;
	(void)lcuOriginX;

	//Luma
	{
		picHeight = inputPicturePtr->height;
		picWidth = inputPicturePtr->width;
		lcuHeight = MIN(MAX_LCU_SIZE, picHeight - lcuOriginY);
		lcuWidth = MIN(MAX_LCU_SIZE, picWidth - lcuOriginX);
		lcuHeight = ((lcuOriginY + MAX_LCU_SIZE >= picHeight) || (lcuOriginY == 0)) ? lcuHeight - 1 : lcuHeight;
		strideIn = inputPicturePtr->strideY;
		inputOriginIndex = inputPicturePtr->originX + lcuOriginX + (inputPicturePtr->originY + lcuOriginY) * inputPicturePtr->strideY;
		ptrIn = &(inputPicturePtr->bufferY[inputOriginIndex]);

		inputOriginIndexPad = denoisedPicturePtr->originX + lcuOriginX + (denoisedPicturePtr->originY + lcuOriginY) * denoisedPicturePtr->strideY;
		strideOut = denoisedPicturePtr->strideY;
		ptrDenoised = &(denoisedPicturePtr->bufferY[inputOriginIndexPad]);
		ptrDenoisedInterm = ptrDenoised;

		noiseOriginIndex = noisePicturePtr->originX + lcuOriginX + noisePicturePtr->originY * noisePicturePtr->strideY;
		ptrNoise = &(noisePicturePtr->bufferY[noiseOriginIndex]);
		ptrNoiseInterm = ptrNoise;

		////Luma
		//a = (p[1] +
		//	p[0 + stride] + 4 * p[1 + stride] + p[2 + stride] +
		//	p[1 + 2 * stride]) / 8;

		top = curr = secondtop = secondcurr = simde_mm256_setzero_si256();

		//for (kk = 0; kk + MAX_LCU_SIZE <= picWidth; kk += MAX_LCU_SIZE)
		{
			for (jj = 0; jj < lcuHeight; jj++)
			{
				if (lcuOriginY == 0)
				{
					if (jj == 0)
					{
						top = simde_mm256_loadu_si256((simde__m256i*)(ptrIn + jj*strideIn));
						secondtop = simde_mm256_loadu_si256((simde__m256i*)(ptrIn + 32 + jj*strideIn));
						curr = simde_mm256_loadu_si256((simde__m256i*)(ptrIn + (1 + jj)*strideIn));
						secondcurr = simde_mm256_loadu_si256((simde__m256i*)((ptrIn + 32) + (1 + jj)*strideIn));
						simde_mm256_storeu_si256((simde__m256i *)(ptrDenoised ), top);
						simde_mm256_storeu_si256((simde__m256i *)(ptrDenoised + 32), secondtop);
						simde_mm256_storeu_si256((simde__m256i *)(ptrNoise ), simde_mm256_setzero_si256());
						simde_mm256_storeu_si256((simde__m256i *)(ptrNoise + 32), simde_mm256_setzero_si256());
					}
					currPrev = simde_mm256_loadu_si256((simde__m256i*)(ptrIn - 1 + ((1 + jj)*strideIn)));
					currNext = simde_mm256_loadu_si256((simde__m256i*)(ptrIn + 1 + ((1 + jj)*strideIn)));
					secondcurrPrev = simde_mm256_loadu_si256((simde__m256i*)((ptrIn + 32) - 1 + ((1 + jj)*strideIn)));
					secondcurrNext = simde_mm256_loadu_si256((simde__m256i*)((ptrIn + 32) + 1 + ((1 + jj)*strideIn)));
					bottom = simde_mm256_loadu_si256((simde__m256i*)((ptrIn ) + (2 + jj)* strideIn));
					secondbottom = simde_mm256_loadu_si256((simde__m256i*)((ptrIn + 32) + (2 + jj)* strideIn));
					ptrDenoisedInterm = ptrDenoised + ((1 + jj)*strideOut);
					ptrNoiseInterm = ptrNoise + ((1 + jj)*strideOut);

				}
				else
				{
					if (jj == 0)
					{
						top = simde_mm256_loadu_si256((simde__m256i*)(ptrIn + jj*strideIn - strideIn));
						secondtop = simde_mm256_loadu_si256((simde__m256i*)(ptrIn + 32 + jj *strideIn - strideIn));
						curr = simde_mm256_loadu_si256((simde__m256i*)(ptrIn + (1 + jj)*strideIn - strideIn));
						secondcurr = simde_mm256_loadu_si256((simde__m256i*)((ptrIn + 32) + (1 + jj)*strideIn - strideIn));
					}
					currPrev = simde_mm256_loadu_si256((simde__m256i*)(ptrIn - 1 + ((1 + jj)*strideIn - strideIn)));
					currNext = simde_mm256_loadu_si256((simde__m256i*)(ptrIn + 1 + ((1 + jj)*strideIn - strideIn)));
					secondcurrPrev = simde_mm256_loadu_si256((simde__m256i*)((ptrIn  + 32) - 1 + ((1 + jj)*strideIn - strideIn)));
					secondcurrNext = simde_mm256_loadu_si256((simde__m256i*)((ptrIn  + 32) + 1 + ((1 + jj)*strideIn - strideIn)));
					bottom = simde_mm256_loadu_si256((simde__m256i*)((ptrIn ) + (2 + jj)* strideIn - strideIn));
					secondbottom = simde_mm256_loadu_si256((simde__m256i*)((ptrIn + 32) + (2 + jj)* strideIn - strideIn));
					ptrDenoisedInterm = ptrDenoised  + ((1 + jj)*strideOut - strideOut);
					ptrNoiseInterm = ptrNoise + jj*strideOut;

				}

				lumaWeakFilter_AVX2_INTRIN(
					top,
					curr,
					bottom,
					currPrev,
					currNext,
					ptrDenoisedInterm,
					ptrNoiseInterm);

				lumaWeakFilter_AVX2_INTRIN(
					secondtop,
					secondcurr,
					secondbottom,
					secondcurrPrev,
					secondcurrNext,
					ptrDenoisedInterm + 32,
					ptrNoiseInterm + 32);

				top = curr;
				curr = bottom;
				secondtop = secondcurr;
				secondcurr = secondbottom;

			}


		}

		lcuHeight = MIN(MAX_LCU_SIZE, picHeight - lcuOriginY);

		for (jj = 0; jj < lcuHeight; jj++){
			for (ii = 0; ii < lcuWidth; ii++){

				if (!((jj>0 || lcuOriginY>0) && (jj<lcuHeight - 1 || lcuOriginY + lcuHeight<picHeight) && (ii>0 || lcuOriginX>0) && (ii + lcuOriginX) < picWidth - 1)){

					ptrDenoised[ii + jj*strideOut] = ptrIn[ii + jj*strideIn];
					ptrNoise[ii + jj*strideOut] = 0;
				}

			}
		}

	}

}
/*******************************************
* noiseExtractLumaStrong
*  strong filter Luma.
*******************************************/
void noiseExtractLumaStrong_AVX2_INTRIN(
	EbPictureBufferDesc_t       *inputPicturePtr,
	EbPictureBufferDesc_t       *denoisedPicturePtr,
	EB_U32                       lcuOriginY,
	EB_U32                       lcuOriginX
	)
{
	EB_U32  ii, jj, kk;
	EB_U32  picHeight, lcuHeight;
	EB_U32  picWidth;
	EB_U32  inputOriginIndex;
	EB_U32  inputOriginIndexPad;

	EB_U8 *ptrIn;
	EB_U32 strideIn;
	EB_U8 *ptrDenoised, *ptrDenoisedInterm;

	EB_U32 strideOut;
	simde__m256i top, curr, bottom, currPrev, currNext, topPrev, topNext, bottomPrev, bottomNext,
		secondtop, secondcurr, secondcurrPrev, secondcurrNext, secondbottom, secondtopPrev, secondtopNext, secondbottomPrev, secondbottomNext;
	(void)lcuOriginX;
	//Luma
	{
		picHeight = inputPicturePtr->height;
		picWidth = inputPicturePtr->width;
		lcuHeight = MIN(MAX_LCU_SIZE, picHeight - lcuOriginY);

		lcuHeight = ((lcuOriginY + MAX_LCU_SIZE >= picHeight) || (lcuOriginY == 0)) ? lcuHeight - 1 : lcuHeight;
		strideIn = inputPicturePtr->strideY;
		inputOriginIndex = inputPicturePtr->originX + (inputPicturePtr->originY + lcuOriginY)* inputPicturePtr->strideY;
		ptrIn = &(inputPicturePtr->bufferY[inputOriginIndex]);

		inputOriginIndexPad = denoisedPicturePtr->originX + (denoisedPicturePtr->originY + lcuOriginY) * denoisedPicturePtr->strideY;
		strideOut = denoisedPicturePtr->strideY;
		ptrDenoised = &(denoisedPicturePtr->bufferY[inputOriginIndexPad]);
		ptrDenoisedInterm = ptrDenoised;


		top = curr = secondtop = secondcurr = topNext = topPrev = currNext = currPrev = secondcurrPrev = secondcurrNext = secondtopPrev = secondtopNext = simde_mm256_setzero_si256();
		for (kk = 0; kk + MAX_LCU_SIZE <= picWidth; kk += MAX_LCU_SIZE)
		{
			for (jj = 0; jj < lcuHeight; jj++)
			{
				if (lcuOriginY == 0)
				{
					if (jj == 0)
					{
						top = simde_mm256_loadu_si256((simde__m256i*)(ptrIn + kk + jj*strideIn));
						secondtop = simde_mm256_loadu_si256((simde__m256i*)(ptrIn + kk + 32 + jj*strideIn));

						curr = simde_mm256_loadu_si256((simde__m256i*)(ptrIn + kk + (1 + jj)*strideIn));
						secondcurr = simde_mm256_loadu_si256((simde__m256i*)((ptrIn + kk + 32) + (1 + jj)*strideIn));

						topPrev = simde_mm256_loadu_si256((simde__m256i*)(ptrIn - 1 + kk + ((jj)*strideIn)));
						secondtopPrev = simde_mm256_loadu_si256((simde__m256i*)(ptrIn - 1 + kk + 32 + ((jj)*strideIn)));

						topNext = simde_mm256_loadu_si256((simde__m256i*)(ptrIn + 1 + kk + ((jj)*strideIn)));
						secondtopNext = simde_mm256_loadu_si256((simde__m256i*)(ptrIn + 1 + kk + 32 + ((jj)*strideIn)));

						currPrev = simde_mm256_loadu_si256((simde__m256i*)(ptrIn - 1 + kk + ((1 + jj)*strideIn)));
						secondcurrPrev = simde_mm256_loadu_si256((simde__m256i*)(ptrIn - 1 + kk + 32 + ((1 + jj)*strideIn)));

						currNext = simde_mm256_loadu_si256((simde__m256i*)(ptrIn + 1 + kk + ((1 + jj)*strideIn)));
						secondcurrNext = simde_mm256_loadu_si256((simde__m256i*)(ptrIn + 1 + kk + 32 + ((1 + jj)*strideIn)));

						simde_mm256_storeu_si256((simde__m256i *)(ptrDenoised + kk), top);
						simde_mm256_storeu_si256((simde__m256i *)(ptrDenoised + kk + 32), secondtop);
					}
					bottomPrev = simde_mm256_loadu_si256((simde__m256i*)(ptrIn - 1 + kk + ((2 + jj)*strideIn)));
					secondbottomPrev = simde_mm256_loadu_si256((simde__m256i*)(ptrIn - 1 + kk + 32 + ((2 + jj)*strideIn)));

					bottomNext = simde_mm256_loadu_si256((simde__m256i*)(ptrIn + 1 + kk + ((2 + jj)*strideIn)));
					secondbottomNext = simde_mm256_loadu_si256((simde__m256i*)(ptrIn + 1 + kk + 32 + ((2 + jj)*strideIn)));

					bottom = simde_mm256_loadu_si256((simde__m256i*)((ptrIn + kk) + (2 + jj)* strideIn));
					secondbottom = simde_mm256_loadu_si256((simde__m256i*)((ptrIn + kk + 32) + (2 + jj)* strideIn));
					ptrDenoisedInterm = ptrDenoised + kk + ((1 + jj)*strideOut);
				}
				else
				{
					if (jj == 0)
					{
						top = simde_mm256_loadu_si256((simde__m256i*)(ptrIn + kk + jj*strideIn - strideIn));
						secondtop = simde_mm256_loadu_si256((simde__m256i*)(ptrIn + kk + 32 + jj*strideIn - strideIn));
						curr = simde_mm256_loadu_si256((simde__m256i*)(ptrIn + kk + (1 + jj)*strideIn - strideIn));
						secondcurr = simde_mm256_loadu_si256((simde__m256i*)((ptrIn + kk + 32) + (1 + jj)*strideIn - strideIn));
						topPrev = simde_mm256_loadu_si256((simde__m256i*)(ptrIn - 1 + kk + ((jj)*strideIn) - strideIn));
						secondtopPrev = simde_mm256_loadu_si256((simde__m256i*)(ptrIn - 1 + kk + 32 + ((jj)*strideIn) - strideIn));

						topNext = simde_mm256_loadu_si256((simde__m256i*)(ptrIn + 1 + kk + ((jj)*strideIn) - strideIn));
						secondtopNext = simde_mm256_loadu_si256((simde__m256i*)(ptrIn + 1 + kk + 32 + ((jj)*strideIn) - strideIn));

						currPrev = simde_mm256_loadu_si256((simde__m256i*)(ptrIn - 1 + kk + ((1 + jj)*strideIn - strideIn)));
						secondcurrPrev = simde_mm256_loadu_si256((simde__m256i*)(ptrIn - 1 + kk + 32 + ((1 + jj)*strideIn - strideIn)));

						currNext = simde_mm256_loadu_si256((simde__m256i*)(ptrIn + 1 + kk + ((1 + jj)*strideIn - strideIn)));
						secondcurrNext = simde_mm256_loadu_si256((simde__m256i*)(ptrIn + 1 + kk + 32 + ((1 + jj)*strideIn - strideIn)));
					}
					bottomPrev = simde_mm256_loadu_si256((simde__m256i*)(ptrIn - 1 + kk + ((2 + jj)*strideIn) - strideIn));
					secondbottomPrev = simde_mm256_loadu_si256((simde__m256i*)(ptrIn - 1 + kk + 32 + ((2 + jj)*strideIn - strideIn)));

					bottomNext = simde_mm256_loadu_si256((simde__m256i*)(ptrIn + 1 + kk + ((2 + jj)*strideIn) - strideIn));
					secondbottomNext = simde_mm256_loadu_si256((simde__m256i*)(ptrIn + 1 + kk + 32 + ((2 + jj)*strideIn - strideIn)));

					bottom = simde_mm256_loadu_si256((simde__m256i*)((ptrIn + kk) + (2 + jj)* strideIn - strideIn));
					secondbottom = simde_mm256_loadu_si256((simde__m256i*)((ptrIn + kk + 32) + (2 + jj)* strideIn - strideIn));

					ptrDenoisedInterm = ptrDenoised + kk + ((1 + jj)*strideOut - strideOut);

				}

				chromaWeakLumaStrongFilter_AVX2_INTRIN(
					top,
					curr,
					bottom,
					currPrev,
					currNext,
					topPrev,
					topNext,
					bottomPrev,
					bottomNext,
					ptrDenoisedInterm);


				chromaWeakLumaStrongFilter_AVX2_INTRIN(
					secondtop,
					secondcurr,
					secondbottom,
					secondcurrPrev,
					secondcurrNext,
					secondtopPrev,
					secondtopNext,
					secondbottomPrev,
					secondbottomNext,
					ptrDenoisedInterm + 32);


				top = curr;
				curr = bottom;
				topPrev = currPrev;
				topNext = currNext;
				currPrev = bottomPrev;
				currNext = bottomNext;
				secondtop = secondcurr;
				secondcurr = secondbottom;
				secondtopPrev = secondcurrPrev;
				secondtopNext = secondcurrNext;
				secondcurrPrev = secondbottomPrev;
				secondcurrNext = secondbottomNext;

			}


		}

		lcuHeight = MIN(MAX_LCU_SIZE, picHeight - lcuOriginY);

		for (jj = 0; jj < lcuHeight; jj++){
			for (ii = 0; ii < picWidth; ii++){

				if (!((jj<lcuHeight - 1 || lcuOriginY + lcuHeight<picHeight) && ii>0 && ii < picWidth - 1)){
					ptrDenoised[ii + jj*strideOut] = ptrIn[ii + jj*strideIn];
				}

			}
		}

	}

}

/*******************************************
* noiseExtractChromaStrong
*  strong filter chroma.
*******************************************/
void noiseExtractChromaStrong_AVX2_INTRIN(
	EbPictureBufferDesc_t       *inputPicturePtr,
	EbPictureBufferDesc_t       *denoisedPicturePtr,
	EB_U32                       lcuOriginY,
	EB_U32                       lcuOriginX
	)
{
	EB_U32  ii, jj, kk;
	EB_U32  picHeight, lcuHeight;
	EB_U32  picWidth;
	EB_U32  inputOriginIndex;
	EB_U32  inputOriginIndexPad;

	EB_U8 *ptrIn, *ptrInCr;
	EB_U32 strideIn, strideInCr;
	EB_U8 *ptrDenoised, *ptrDenoisedInterm, *ptrDenoisedCr, *ptrDenoisedIntermCr;

	EB_U32 strideOut, strideOutCr;

    EB_U32 colorFormat=inputPicturePtr->colorFormat;
    EB_U16 subWidthCMinus1 = (colorFormat==EB_YUV444?1:2)-1;
    EB_U16 subHeightCMinus1 = (colorFormat>=EB_YUV422?1:2)-1;

	simde__m256i top, curr, bottom, currPrev, currNext, topPrev, topNext, bottomPrev, bottomNext,
		topCr, currCr, bottomCr, currPrevCr, currNextCr, topPrevCr, topNextCr, bottomPrevCr, bottomNextCr;
	(void)lcuOriginX;
	{
		picHeight = inputPicturePtr->height >> subHeightCMinus1;
		picWidth = inputPicturePtr->width >> subWidthCMinus1;
        lcuHeight = MIN(MAX_LCU_SIZE >> subHeightCMinus1, picHeight - lcuOriginY);

        lcuHeight = ((lcuOriginY + (MAX_LCU_SIZE >> subHeightCMinus1) >= picHeight) || (lcuOriginY == 0)) ? lcuHeight - 1 : lcuHeight;

		strideIn = inputPicturePtr->strideCb;
        inputOriginIndex = (inputPicturePtr->originX >> subWidthCMinus1) + ((inputPicturePtr->originY >> subHeightCMinus1) + lcuOriginY)  * inputPicturePtr->strideCb;
		ptrIn = &(inputPicturePtr->bufferCb[inputOriginIndex]);

        inputOriginIndexPad = (denoisedPicturePtr->originX >> subWidthCMinus1) + ((denoisedPicturePtr->originY >> subHeightCMinus1) + lcuOriginY)  * denoisedPicturePtr->strideCb;
		strideOut = denoisedPicturePtr->strideCb;
		ptrDenoised = &(denoisedPicturePtr->bufferCb[inputOriginIndexPad]);
		ptrDenoisedInterm = ptrDenoised;

		strideInCr = inputPicturePtr->strideCr;
        inputOriginIndex = (inputPicturePtr->originX >> subWidthCMinus1) + ((inputPicturePtr->originY >> subHeightCMinus1) + lcuOriginY)  * inputPicturePtr->strideCr;
		ptrInCr = &(inputPicturePtr->bufferCr[inputOriginIndex]);

        inputOriginIndexPad = (denoisedPicturePtr->originX >> subWidthCMinus1) + ((denoisedPicturePtr->originY >> subHeightCMinus1) + lcuOriginY)  * denoisedPicturePtr->strideCr;
		strideOutCr = denoisedPicturePtr->strideCr;
		ptrDenoisedCr = &(denoisedPicturePtr->bufferCr[inputOriginIndexPad]);
		ptrDenoisedIntermCr = ptrDenoisedCr;
		////Chroma
		//a = (4 * p[0] + 4 * p[1] + 4 * p[2] +
		//	4 * p[0 + stride] + 4 * p[1 + stride] + 4 * p[2 + stride] +
		//	4 * p[0 + 2 * stride] + 4 * p[1 + 2 * stride] + 4 * p[2 + 2 * stride]) / 36;

		top = curr = topNext = topPrev = currNext = currPrev = topCr = currCr = topNextCr = topPrevCr = currNextCr = currPrevCr =  simde_mm256_setzero_si256();

		for (kk = 0; kk + MAX_LCU_SIZE / 2 <= picWidth; kk += MAX_LCU_SIZE / 2)
		{

			for (jj = 0; jj < lcuHeight; jj++)
			{
				if (lcuOriginY == 0)
				{
					if (jj == 0)
					{
						top = simde_mm256_loadu_si256((simde__m256i*)(ptrIn + kk + jj*strideIn));
						curr = simde_mm256_loadu_si256((simde__m256i*)(ptrIn + kk + (1 + jj)*strideIn));
						topPrev = simde_mm256_loadu_si256((simde__m256i*)(ptrIn - 1 + kk + ((jj)*strideIn)));
						topNext = simde_mm256_loadu_si256((simde__m256i*)(ptrIn + 1 + kk + ((jj)*strideIn)));
						currPrev = simde_mm256_loadu_si256((simde__m256i*)(ptrIn - 1 + kk + ((1 + jj)*strideIn)));
						currNext = simde_mm256_loadu_si256((simde__m256i*)(ptrIn + 1 + kk + ((1 + jj)*strideIn)));
						topCr = simde_mm256_loadu_si256((simde__m256i*)(ptrInCr + kk + jj*strideInCr));
						currCr = simde_mm256_loadu_si256((simde__m256i*)(ptrInCr + kk + (1 + jj)*strideInCr));
						topPrevCr = simde_mm256_loadu_si256((simde__m256i*)(ptrInCr - 1 + kk + ((jj)*strideInCr)));
						topNextCr = simde_mm256_loadu_si256((simde__m256i*)(ptrInCr + 1 + kk + ((jj)*strideInCr)));
						currPrevCr = simde_mm256_loadu_si256((simde__m256i*)(ptrInCr - 1 + kk + ((1 + jj)*strideInCr)));
						currNextCr = simde_mm256_loadu_si256((simde__m256i*)(ptrInCr + 1 + kk + ((1 + jj)*strideInCr)));
						simde_mm256_storeu_si256((simde__m256i *)(ptrDenoised + kk), top);
						simde_mm256_storeu_si256((simde__m256i *)(ptrDenoisedCr + kk), topCr);
					}
					bottomPrev = simde_mm256_loadu_si256((simde__m256i*)(ptrIn - 1 + kk + ((2 + jj)*strideIn)));
					bottomNext = simde_mm256_loadu_si256((simde__m256i*)(ptrIn + 1 + kk + ((2 + jj)*strideIn)));
					bottom = simde_mm256_loadu_si256((simde__m256i*)((ptrIn + kk) + (2 + jj)* strideIn));
					bottomPrevCr = simde_mm256_loadu_si256((simde__m256i*)(ptrInCr - 1 + kk + ((2 + jj)*strideInCr)));
					bottomNextCr = simde_mm256_loadu_si256((simde__m256i*)(ptrInCr + 1 + kk + ((2 + jj)*strideInCr)));
					bottomCr = simde_mm256_loadu_si256((simde__m256i*)((ptrInCr + kk) + (2 + jj)* strideInCr));
					ptrDenoisedInterm = ptrDenoised + kk + ((1 + jj)*strideOut);
					ptrDenoisedIntermCr = ptrDenoisedCr + kk + ((1 + jj)*strideOutCr);
				}
				else
				{
					if (jj == 0)
					{
						top = simde_mm256_loadu_si256((simde__m256i*)(ptrIn + kk + jj*strideIn - strideIn));
						curr = simde_mm256_loadu_si256((simde__m256i*)(ptrIn + kk + (1 + jj)*strideIn - strideIn));
						topPrev = simde_mm256_loadu_si256((simde__m256i*)(ptrIn - 1 + kk + ((jj)*strideIn) - strideIn));
						topNext = simde_mm256_loadu_si256((simde__m256i*)(ptrIn + 1 + kk + ((jj)*strideIn) - strideIn));
						currPrev = simde_mm256_loadu_si256((simde__m256i*)(ptrIn - 1 + kk + ((1 + jj)*strideIn - strideIn)));
						currNext = simde_mm256_loadu_si256((simde__m256i*)(ptrIn + 1 + kk + ((1 + jj)*strideIn - strideIn)));
						topCr = simde_mm256_loadu_si256((simde__m256i*)(ptrInCr + kk + jj*strideInCr - strideInCr));
						currCr = simde_mm256_loadu_si256((simde__m256i*)(ptrInCr + kk + (1 + jj)*strideInCr - strideInCr));
						topPrevCr = simde_mm256_loadu_si256((simde__m256i*)(ptrInCr - 1 + kk + ((jj)*strideInCr) - strideInCr));
						topNextCr = simde_mm256_loadu_si256((simde__m256i*)(ptrInCr + 1 + kk + ((jj)*strideInCr) - strideInCr));
						currPrevCr = simde_mm256_loadu_si256((simde__m256i*)(ptrInCr - 1 + kk + ((1 + jj)*strideInCr - strideInCr)));
						currNextCr = simde_mm256_loadu_si256((simde__m256i*)(ptrInCr + 1 + kk + ((1 + jj)*strideInCr - strideInCr)));
					}
					bottomPrev = simde_mm256_loadu_si256((simde__m256i*)(ptrIn - 1 + kk + ((2 + jj)*strideIn) - strideIn));
					bottomNext = simde_mm256_loadu_si256((simde__m256i*)(ptrIn + 1 + kk + ((2 + jj)*strideIn) - strideIn));
					bottom = simde_mm256_loadu_si256((simde__m256i*)((ptrIn + kk) + (2 + jj)* strideIn - strideIn));
					ptrDenoisedInterm = ptrDenoised + kk + ((1 + jj)*strideOut - strideOut);
					bottomPrevCr = simde_mm256_loadu_si256((simde__m256i*)(ptrInCr - 1 + kk + ((2 + jj)*strideInCr) - strideInCr));
					bottomNextCr = simde_mm256_loadu_si256((simde__m256i*)(ptrInCr + 1 + kk + ((2 + jj)*strideInCr) - strideInCr));
					bottomCr = simde_mm256_loadu_si256((simde__m256i*)((ptrInCr + kk) + (2 + jj)* strideInCr - strideInCr));
					ptrDenoisedIntermCr = ptrDenoisedCr + kk + ((1 + jj)*strideOutCr - strideOutCr);


				}

				ChromaStrong_AVX2_INTRIN(
					top,
					curr,
					bottom,
					currPrev,
					currNext,
					topPrev,
					topNext,
					bottomPrev,
					bottomNext,
					ptrDenoisedInterm);

				ChromaStrong_AVX2_INTRIN(
					topCr,
					currCr,
					bottomCr,
					currPrevCr,
					currNextCr,
					topPrevCr,
					topNextCr,
					bottomPrevCr,
					bottomNextCr,
					ptrDenoisedIntermCr);

				top = curr;
				curr = bottom;
				topPrev = currPrev;
				topNext = currNext;
				currPrev = bottomPrev;
				currNext = bottomNext;
				topCr = currCr;
				currCr = bottomCr;
				topPrevCr = currPrevCr;
				topNextCr = currNextCr;
				currPrevCr = bottomPrevCr;
				currNextCr = bottomNextCr;

			}


		}

        lcuHeight = MIN(MAX_LCU_SIZE >> subHeightCMinus1, picHeight - lcuOriginY);

		for (jj = 0; jj < lcuHeight; jj++){
			for (ii = 0; ii < picWidth; ii++){

				if (!((jj<lcuHeight - 1 || (lcuOriginY + lcuHeight)<picHeight) && ii>0 && ii < picWidth - 1)){
					ptrDenoised[ii + jj*strideOut] = ptrIn[ii + jj*strideIn];
					ptrDenoisedCr[ii + jj*strideOut] = ptrInCr[ii + jj*strideIn];
				}

			}
		}
	}


}

/*******************************************
* noiseExtractChromaWeak
*  weak filter chroma.
*******************************************/
void noiseExtractChromaWeak_AVX2_INTRIN(
	EbPictureBufferDesc_t       *inputPicturePtr,
	EbPictureBufferDesc_t       *denoisedPicturePtr,
	EB_U32                       lcuOriginY,
	EB_U32                       lcuOriginX
	)
{
	EB_U32  ii, jj, kk;
	EB_U32  picHeight, lcuHeight;
	EB_U32  picWidth;
	EB_U32  inputOriginIndex;
	EB_U32  inputOriginIndexPad;

	EB_U8 *ptrIn, *ptrInCr;
	EB_U32 strideIn, strideInCr;
	EB_U8 *ptrDenoised, *ptrDenoisedInterm, *ptrDenoisedCr, *ptrDenoisedIntermCr;

	EB_U32 strideOut, strideOutCr;

    EB_U32 colorFormat=inputPicturePtr->colorFormat;
    EB_U16 subWidthCMinus1 = (colorFormat==EB_YUV444?1:2)-1;
    EB_U16 subHeightCMinus1 = (colorFormat>=EB_YUV422?1:2)-1;

	simde__m256i top, curr, bottom, currPrev, currNext, topPrev, topNext, bottomPrev, bottomNext,
		topCr, currCr, bottomCr, currPrevCr, currNextCr, topPrevCr, topNextCr, bottomPrevCr, bottomNextCr;
	(void)lcuOriginX;

	////gaussian matrix(Chroma)
	//a = (1 * p[0] + 2 * p[1] + 1 * p[2] +
	//	2 * p[0 + stride] + 4 * p[1 + stride] + 2 * p[2 + stride] +
	//	1 * p[0 + 2 * stride] + 2 * p[1 + 2 * stride] + 1 * p[2 + 2 * stride]) / 16;

	{
        picHeight = inputPicturePtr->height >> subHeightCMinus1;
        picWidth = inputPicturePtr->width >> subWidthCMinus1 ;


        lcuHeight = MIN(MAX_LCU_SIZE >> subHeightCMinus1, picHeight - lcuOriginY);

        lcuHeight = ((lcuOriginY + (MAX_LCU_SIZE >> subHeightCMinus1) >= picHeight) || (lcuOriginY == 0)) ? lcuHeight - 1 : lcuHeight;
		strideIn = inputPicturePtr->strideCb;
        inputOriginIndex = (inputPicturePtr->originX >> subWidthCMinus1) + ((inputPicturePtr->originY >> subHeightCMinus1) + lcuOriginY)* inputPicturePtr->strideCb;
		ptrIn = &(inputPicturePtr->bufferCb[inputOriginIndex]);

        inputOriginIndexPad = (denoisedPicturePtr->originX >> subWidthCMinus1) + ((denoisedPicturePtr->originY >> subHeightCMinus1) + lcuOriginY)* denoisedPicturePtr->strideCb;
		strideOut = denoisedPicturePtr->strideCb;
		ptrDenoised = &(denoisedPicturePtr->bufferCb[inputOriginIndexPad]);
		ptrDenoisedInterm = ptrDenoised;


		strideInCr = inputPicturePtr->strideCr;
        inputOriginIndex = (inputPicturePtr->originX >> subWidthCMinus1) + ((inputPicturePtr->originY >> subHeightCMinus1) + lcuOriginY)  * inputPicturePtr->strideCr;
		ptrInCr = &(inputPicturePtr->bufferCr[inputOriginIndex]);

        inputOriginIndexPad = (denoisedPicturePtr->originX >> subWidthCMinus1) + ((denoisedPicturePtr->originY >> subHeightCMinus1) + lcuOriginY)  * denoisedPicturePtr->strideCr;
		strideOutCr = denoisedPicturePtr->strideCr;
		ptrDenoisedCr = &(denoisedPicturePtr->bufferCr[inputOriginIndexPad]);
		ptrDenoisedIntermCr = ptrDenoisedCr;

		top = curr = topNext = topPrev = currNext = currPrev = topCr = currCr = topNextCr = topPrevCr = currNextCr = currPrevCr = simde_mm256_setzero_si256();
		for (kk = 0; kk + MAX_LCU_SIZE / 2 <= picWidth; kk += MAX_LCU_SIZE / 2)
		{

			for (jj = 0; jj < lcuHeight; jj++)
			{
				if (lcuOriginY == 0)
				{
					if (jj == 0)
					{
						top = simde_mm256_loadu_si256((simde__m256i*)(ptrIn + kk + jj*strideIn));
						curr = simde_mm256_loadu_si256((simde__m256i*)(ptrIn + kk + (1 + jj)*strideIn));
						topPrev = simde_mm256_loadu_si256((simde__m256i*)(ptrIn - 1 + kk + ((jj)*strideIn)));
						topNext = simde_mm256_loadu_si256((simde__m256i*)(ptrIn + 1 + kk + ((jj)*strideIn)));
						currPrev = simde_mm256_loadu_si256((simde__m256i*)(ptrIn - 1 + kk + ((1 + jj)*strideIn)));
						currNext = simde_mm256_loadu_si256((simde__m256i*)(ptrIn + 1 + kk + ((1 + jj)*strideIn)));
						simde_mm256_storeu_si256((simde__m256i *)(ptrDenoised + kk), top);
						topCr = simde_mm256_loadu_si256((simde__m256i*)(ptrInCr + kk + jj*strideInCr));
						currCr = simde_mm256_loadu_si256((simde__m256i*)(ptrInCr + kk + (1 + jj)*strideInCr));
						topPrevCr = simde_mm256_loadu_si256((simde__m256i*)(ptrInCr - 1 + kk + ((jj)*strideInCr)));
						topNextCr = simde_mm256_loadu_si256((simde__m256i*)(ptrInCr + 1 + kk + ((jj)*strideInCr)));
						currPrevCr = simde_mm256_loadu_si256((simde__m256i*)(ptrInCr - 1 + kk + ((1 + jj)*strideInCr)));
						currNextCr = simde_mm256_loadu_si256((simde__m256i*)(ptrInCr + 1 + kk + ((1 + jj)*strideInCr)));
						simde_mm256_storeu_si256((simde__m256i *)(ptrDenoisedCr + kk), topCr);
					}
					bottomPrev = simde_mm256_loadu_si256((simde__m256i*)(ptrIn - 1 + kk + ((2 + jj)*strideIn)));
					bottomNext = simde_mm256_loadu_si256((simde__m256i*)(ptrIn + 1 + kk + ((2 + jj)*strideIn)));
					bottom = simde_mm256_loadu_si256((simde__m256i*)((ptrIn + kk) + (2 + jj)* strideIn));
					ptrDenoisedInterm = ptrDenoised + kk + ((1 + jj)*strideOut);
					bottomPrevCr = simde_mm256_loadu_si256((simde__m256i*)(ptrInCr - 1 + kk + ((2 + jj)*strideInCr)));
					bottomNextCr = simde_mm256_loadu_si256((simde__m256i*)(ptrInCr + 1 + kk + ((2 + jj)*strideInCr)));
					bottomCr = simde_mm256_loadu_si256((simde__m256i*)((ptrInCr + kk) + (2 + jj)* strideInCr));
					ptrDenoisedIntermCr = ptrDenoisedCr + kk + ((1 + jj)*strideOutCr);
				}
				else
				{
					if (jj == 0)
					{
						top = simde_mm256_loadu_si256((simde__m256i*)(ptrIn + kk + jj*strideIn - strideIn));
						curr = simde_mm256_loadu_si256((simde__m256i*)(ptrIn + kk + (1 + jj)*strideIn - strideIn));
						topPrev = simde_mm256_loadu_si256((simde__m256i*)(ptrIn - 1 + kk + ((jj)*strideIn) - strideIn));
						topNext = simde_mm256_loadu_si256((simde__m256i*)(ptrIn + 1 + kk + ((jj)*strideIn) - strideIn));
						currPrev = simde_mm256_loadu_si256((simde__m256i*)(ptrIn - 1 + kk + ((1 + jj)*strideIn - strideIn)));
						currNext = simde_mm256_loadu_si256((simde__m256i*)(ptrIn + 1 + kk + ((1 + jj)*strideIn - strideIn)));
						topCr = simde_mm256_loadu_si256((simde__m256i*)(ptrInCr + kk + jj*strideInCr - strideInCr));
						currCr = simde_mm256_loadu_si256((simde__m256i*)(ptrInCr + kk + (1 + jj)*strideInCr - strideInCr));
						topPrevCr = simde_mm256_loadu_si256((simde__m256i*)(ptrInCr - 1 + kk + ((jj)*strideInCr) - strideInCr));
						topNextCr = simde_mm256_loadu_si256((simde__m256i*)(ptrInCr + 1 + kk + ((jj)*strideInCr) - strideInCr));
						currPrevCr = simde_mm256_loadu_si256((simde__m256i*)(ptrInCr - 1 + kk + ((1 + jj)*strideInCr - strideInCr)));
						currNextCr = simde_mm256_loadu_si256((simde__m256i*)(ptrInCr + 1 + kk + ((1 + jj)*strideInCr - strideInCr)));
					}
					bottomPrev = simde_mm256_loadu_si256((simde__m256i*)(ptrIn - 1 + kk + ((2 + jj)*strideIn) - strideIn));
					bottomNext = simde_mm256_loadu_si256((simde__m256i*)(ptrIn + 1 + kk + ((2 + jj)*strideIn) - strideIn));
					bottom = simde_mm256_loadu_si256((simde__m256i*)((ptrIn + kk) + (2 + jj)* strideIn - strideIn));
					ptrDenoisedInterm = ptrDenoised + kk + ((1 + jj)*strideOut - strideOut);
					bottomPrevCr = simde_mm256_loadu_si256((simde__m256i*)(ptrInCr - 1 + kk + ((2 + jj)*strideInCr) - strideInCr));
					bottomNextCr = simde_mm256_loadu_si256((simde__m256i*)(ptrInCr + 1 + kk + ((2 + jj)*strideInCr) - strideInCr));
					bottomCr = simde_mm256_loadu_si256((simde__m256i*)((ptrInCr + kk) + (2 + jj)* strideInCr - strideInCr));
					ptrDenoisedIntermCr = ptrDenoisedCr + kk + ((1 + jj)*strideOutCr - strideOutCr);

				}

				chromaWeakLumaStrongFilter_AVX2_INTRIN(
					top,
					curr,
					bottom,
					currPrev,
					currNext,
					topPrev,
					topNext,
					bottomPrev,
					bottomNext,
					ptrDenoisedInterm);

				chromaWeakLumaStrongFilter_AVX2_INTRIN(
					topCr,
					currCr,
					bottomCr,
					currPrevCr,
					currNextCr,
					topPrevCr,
					topNextCr,
					bottomPrevCr,
					bottomNextCr,
					ptrDenoisedIntermCr);


				top = curr;
				curr = bottom;
				topPrev = currPrev;
				topNext = currNext;
				currPrev = bottomPrev;
				currNext = bottomNext;
				topCr = currCr;
				currCr = bottomCr;
				topPrevCr = currPrevCr;
				topNextCr = currNextCr;
				currPrevCr = bottomPrevCr;
				currNextCr = bottomNextCr;

			}


		}


        lcuHeight = MIN(MAX_LCU_SIZE >> subHeightCMinus1, picHeight - lcuOriginY);
		for (jj = 0; jj < lcuHeight; jj++){
			for (ii = 0; ii < picWidth; ii++){

				if (!((jj<lcuHeight - 1 || (lcuOriginY + lcuHeight)<picHeight) && ii>0 && ii < picWidth - 1)){
					ptrDenoised[ii + jj*strideOut] = ptrIn[ii + jj*strideIn];
					ptrDenoisedCr[ii + jj*strideOut] = ptrInCr[ii + jj*strideInCr];
				}

			}
		}
	}


}
