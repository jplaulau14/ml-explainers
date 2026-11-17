from collections.abc import Callable
from dataclasses import dataclass

from tiling_explainer.shapes import check_size

Dims = tuple[int, int]

NAIVE = r"""
extern "C" __global__ void naive(const float* A, const float* B, float* C, int N) {
  int row = blockIdx.y * 32 + threadIdx.y;
  int col = blockIdx.x * 32 + threadIdx.x;
  float acc = 0.f;
  for (int k = 0; k < N; ++k) acc += A[row * N + k] * B[k * N + col];
  C[row * N + col] = acc;
}
"""

TILE32 = r"""
#define T 32
extern "C" __global__ void tile32(const float* A, const float* B, float* C, int N) {
  __shared__ float As[T][T];
  __shared__ float Bs[T][T];
  int tx = threadIdx.x, ty = threadIdx.y;
  int row = blockIdx.y * T + ty, col = blockIdx.x * T + tx;
  float acc = 0.f;
  for (int t = 0; t < N; t += T) {
    As[ty][tx] = A[row * N + t + tx];
    Bs[ty][tx] = B[(t + ty) * N + col];
    __syncthreads();
    #pragma unroll
    for (int k = 0; k < T; ++k) acc += As[ty][k] * Bs[k][tx];
    __syncthreads();
  }
  C[row * N + col] = acc;
}
"""

REG4X4 = r"""
extern "C" __global__ void __launch_bounds__(256)
reg4x4(const float* A, const float* B, float* C, int N) {
  __shared__ float As[16][64];
  __shared__ float Bs[16][64];
  const int tx = threadIdx.x, ty = threadIdx.y;
  const int tid = ty * 16 + tx;
  const int bm = blockIdx.y * 64, bn = blockIdx.x * 64;
  float acc[4][4] = {{0.f}};
  for (int t = 0; t < N; t += 16) {
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
      int idx = tid + i * 256;
      int r = idx / 16, c = idx % 16;
      As[c][r] = A[(bm + r) * N + t + c];
      int rb = idx / 64, cb = idx % 64;
      Bs[rb][cb] = B[(t + rb) * N + bn + cb];
    }
    __syncthreads();
    #pragma unroll
    for (int k = 0; k < 16; ++k) {
      float a[4], b[4];
      #pragma unroll
      for (int i = 0; i < 4; ++i) a[i] = As[k][ty * 4 + i];
      #pragma unroll
      for (int j = 0; j < 4; ++j) b[j] = Bs[k][tx * 4 + j];
      #pragma unroll
      for (int i = 0; i < 4; ++i)
        #pragma unroll
        for (int j = 0; j < 4; ++j) acc[i][j] += a[i] * b[j];
    }
    __syncthreads();
  }
  for (int i = 0; i < 4; ++i)
    for (int j = 0; j < 4; ++j) C[(bm + ty * 4 + i) * N + bn + tx * 4 + j] = acc[i][j];
}
"""

REG8X8 = r"""
#define BM 128
#define BN 128
#define BK 8
#define TM 8
#define TN 8
extern "C" __global__ void __launch_bounds__(256)
reg8x8(const float* A, const float* B, float* C, int N) {
  __shared__ float As[BK * BM];
  __shared__ float Bs[BK * BN];
  const int tid = threadIdx.x;
  const int threadCol = tid % (BN / TN), threadRow = tid / (BN / TN);
  A += blockIdx.y * BM * N;
  B += blockIdx.x * BN;
  C += blockIdx.y * BM * N + blockIdx.x * BN;
  const int innerRowA = tid / BK, innerColA = tid % BK, strideA = 256 / BK;
  const int innerRowB = tid / BN, innerColB = tid % BN, strideB = 256 / BN;
  float acc[TM * TN] = {0.f};
  float regM[TM], regN[TN];
  for (int bk = 0; bk < N; bk += BK) {
    for (int o = 0; o < BM; o += strideA)
      As[innerColA * BM + innerRowA + o] = A[(innerRowA + o) * N + innerColA];
    for (int o = 0; o < BK; o += strideB)
      Bs[(innerRowB + o) * BN + innerColB] = B[(innerRowB + o) * N + innerColB];
    __syncthreads();
    A += BK; B += BK * N;
    #pragma unroll
    for (int d = 0; d < BK; ++d) {
      #pragma unroll
      for (int i = 0; i < TM; ++i) regM[i] = As[d * BM + threadRow * TM + i];
      #pragma unroll
      for (int i = 0; i < TN; ++i) regN[i] = Bs[d * BN + threadCol * TN + i];
      #pragma unroll
      for (int m = 0; m < TM; ++m)
        #pragma unroll
        for (int n = 0; n < TN; ++n) acc[m * TN + n] += regM[m] * regN[n];
    }
    __syncthreads();
  }
  for (int m = 0; m < TM; ++m)
    for (int n = 0; n < TN; ++n)
      C[(threadRow * TM + m) * N + threadCol * TN + n] = acc[m * TN + n];
}
"""


@dataclass(frozen=True)
class CudaKernel:
    kernel: str
    source: str
    multiple: int
    grid: Callable[[int], Dims]
    block: Dims

    def launch(self, n: int) -> tuple[Dims, Dims]:
        check_size(self.kernel, self.multiple, n)
        return self.grid(n), self.block


KERNELS = {
    "naive": CudaKernel("naive", NAIVE, 32, lambda n: (n // 32, n // 32), (32, 32)),
    "tile32": CudaKernel("tile32", TILE32, 32, lambda n: (n // 32, n // 32), (32, 32)),
    "reg4x4": CudaKernel("reg4x4", REG4X4, 64, lambda n: (n // 64, n // 64), (16, 16)),
    "reg8x8": CudaKernel("reg8x8", REG8X8, 128, lambda n: (n // 128, n // 128), (256, 1)),
}
