// Compile Sputnik's SpMM translation unit as CUDA (torch's cpp_extension keys the language
// off the file extension, and Sputnik ships it as .cu.cc which is treated as host C++).
#include "sputnik/spmm/cuda_spmm.cu.cc"
