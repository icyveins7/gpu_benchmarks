#pragma once

#include <vector>
#include <stdexcept>


template <typename Tin, typename Tmat, typename Tout>
Tout convExplicitlyAtPoint(
  const std::vector<Tmat>& mat,
  const int matLength,
  const Tin *in,
  const int inHeight, const int inWidth,
  const int x, const int y
){
  Tout v = 0;
  for (int i = 0; i < matLength; i++){
    int iy = y - matLength / 2 + i;
    if (iy < 0 || iy >= inHeight){
      continue;
    }
    for (int j = 0; j < matLength; j++){
      int jx = x - matLength / 2 + j;
      if (jx < 0 || jx >= inWidth){
        continue;
      }
      v += in[iy * inWidth + jx] * mat[i * matLength + j];
    }
  }
  return v;
}

template <typename Tin, typename Tmat, typename Tout>
std::vector<Tout> convExplicitly(
  const std::vector<Tmat>& mat,
  const int matLength,
  const Tin *in,
  const int inHeight, const int inWidth
){
  if (matLength % 2 == 0){
    throw std::runtime_error("matLength must be odd");
  }

  std::vector<Tout> out(inHeight * inWidth);
  for (int y = 0; y < inHeight; y++){
    for (int x = 0; x < inWidth; x++){
      out.at(y * inWidth + x) = convExplicitlyAtPoint<Tin, Tmat, Tout>(
        mat,
        matLength,
        in,
        inHeight,
        inWidth,
        x,
        y
      );
    }
  }

  return out;
}

template <typename Tin, typename Tmat, typename Tout, typename Rule>
std::vector<Tout> convExplicitlyWithRule(
  const std::vector<std::vector<Tmat>>& mats,
  const std::vector<int> & matLengths,
  const Tin *in,
  const int inHeight, const int inWidth,
  const Rule rule
){
  std::vector<Tout> out(inHeight * inWidth);
  for (int y = 0; y < inHeight; y++){
    for (int x = 0; x < inWidth; x++){
      int fIdx = rule.getFilterIndex(y, x);
      out.at(y * inWidth + x) = convExplicitlyAtPoint<Tin, Tmat, Tout>(
        mats.at(fIdx),
        matLengths.at(fIdx),
        in,
        inHeight,
        inWidth,
        x,
        y
      );
    }
  }

  return out;
}
