//  Copyright 2013 Google Inc. All Rights Reserved.
//
//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.

//  modified by chenbingjin
//  @2017/12/27 15:00:01

#include <stdio.h>
#include <string.h>
#include <math.h>
#if __APPLE__
#include <stdlib.h>
#else
#include <malloc.h>
#endif

const long long max_size = 2000;         // max length of strings
const long long max_w = 50;              // max length of vocabulary entries

int main(int argc, char **argv) {
  FILE *f;
  FILE *fo;
  char file_name[max_size];
  char output_file[max_size];
  float len;
  long long words, size, a, b;
  float *M;
  char *vocab;
  if (argc < 3) {
    printf("Usage: ./binary_translation <BINFILE> <OUTFILE>\nwhere BINFILE contains word projections in the BINARY FORMAT\n");
    return 0;
  }
  strcpy(file_name, argv[1]);
  strcpy(output_file, argv[2]);
  f = fopen(file_name, "rb");
  if (f == NULL) {
    printf("Input file not found\n");
    return -1;
  }
  fscanf(f, "%lld", &words);
  fscanf(f, "%lld", &size);
  vocab = (char *)malloc((long long)words * max_w * sizeof(char));
  M = (float *)malloc((long long)words * (long long)size * sizeof(float));
  if (M == NULL) {
    printf("Cannot allocate memory: %lld MB    %lld  %lld\n", (long long)words * size * sizeof(float) / 1048576, words, size);
    return -1;
  }
  fo = fopen(output_file, "wb");
  for (b = 0; b < words; b++) {
    // read word name
    a = 0;
    while (1) {
      vocab[b * max_w + a] = fgetc(f);
      if (feof(f) || (vocab[b * max_w + a] == ' ')) break;
      if ((a < max_w) && (vocab[b * max_w + a] != '\n')) a++;
    }
    vocab[b * max_w + a] = 0;
    // read word vector and normalized vector
    for (a = 0; a < size; a++) fread(&M[a + b * size], sizeof(float), 1, f);
    len = 0;
    for (a = 0; a < size; a++) len += M[a + b * size] * M[a + b * size];
    len = sqrt(len);
    for (a = 0; a < size; a++) M[a + b * size] /= len;
    // save word vector in origin txt
    fprintf(fo, "%s ", &vocab[b * max_w]);
    for (a = 0; a < size; a++) fprintf(fo, "%lf ", M[a + b * size]);
    fprintf(fo, "\n");
  }
  fclose(f);
  fclose(fo);
  return 0;
}
