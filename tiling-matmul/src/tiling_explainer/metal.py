from collections.abc import Callable
from dataclasses import dataclass

from tiling_explainer.shapes import check_size

Dims = tuple[int, int, int]

NAIVE = """
    uint col = thread_position_in_grid.x, row = thread_position_in_grid.y;
    float acc = 0.0f;
    for (int k = 0; k < N; ++k) acc += A[row * N + k] * B[k * N + col];
    C[row * N + col] = acc;
"""

TILE32 = """
    threadgroup float As[T][T];
    threadgroup float Bs[T][T];
    uint tx = thread_position_in_threadgroup.x, ty = thread_position_in_threadgroup.y;
    uint col = thread_position_in_grid.x, row = thread_position_in_grid.y;
    float acc = 0.0f;
    for (int t = 0; t < N; t += T) {
        As[ty][tx] = A[row * N + t + tx];
        Bs[ty][tx] = B[(t + ty) * N + col];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int k = 0; k < T; ++k) acc += As[ty][k] * Bs[k][tx];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    C[row * N + col] = acc;
"""

REG4X4 = """
    threadgroup float As[16][64];
    threadgroup float Bs[16][64];
    uint tx = thread_position_in_threadgroup.x, ty = thread_position_in_threadgroup.y;
    uint tid = ty * 16 + tx;
    uint bm = threadgroup_position_in_grid.y * 64, bn = threadgroup_position_in_grid.x * 64;
    float acc[4][4] = {{0.0f}};
    for (int t = 0; t < N; t += 16) {
        for (int i = 0; i < 4; ++i) {
            uint idx = tid + i * 256;
            uint r = idx / 16, c = idx % 16;
            As[c][r] = A[(bm + r) * N + t + c];
            uint rb = idx / 64, cb = idx % 64;
            Bs[rb][cb] = B[(t + rb) * N + bn + cb];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int k = 0; k < 16; ++k) {
            float a[4], b[4];
            for (int i = 0; i < 4; ++i) a[i] = As[k][ty * 4 + i];
            for (int j = 0; j < 4; ++j) b[j] = Bs[k][tx * 4 + j];
            for (int i = 0; i < 4; ++i)
                for (int j = 0; j < 4; ++j) acc[i][j] += a[i] * b[j];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j) C[(bm + ty * 4 + i) * N + bn + tx * 4 + j] = acc[i][j];
"""

REG8X8_HEADER = """
#define BM 128
#define BN 128
#define BK 8
#define TM 8
#define TN 8
"""

REG8X8 = """
    threadgroup float As[BK * BM];
    threadgroup float Bs[BK * BN];
    const int tid = thread_position_in_threadgroup.x;
    const int threadCol = tid % (BN / TN), threadRow = tid / (BN / TN);
    const device float* Ap = A + threadgroup_position_in_grid.y * BM * N;
    const device float* Bp = B + threadgroup_position_in_grid.x * BN;
    device float* Cp = C + threadgroup_position_in_grid.y * BM * N
        + threadgroup_position_in_grid.x * BN;
    const int innerRowA = tid / BK, innerColA = tid % BK, strideA = 256 / BK;
    const int innerRowB = tid / BN, innerColB = tid % BN, strideB = 256 / BN;
    float acc[TM * TN] = {0.0f};
    float regM[TM], regN[TN];
    for (int bk = 0; bk < N; bk += BK) {
        for (int o = 0; o < BM; o += strideA)
            As[innerColA * BM + innerRowA + o] = Ap[(innerRowA + o) * N + innerColA];
        for (int o = 0; o < BK; o += strideB)
            Bs[(innerRowB + o) * BN + innerColB] = Bp[(innerRowB + o) * N + innerColB];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        Ap += BK; Bp += BK * N;
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
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (int m = 0; m < TM; ++m)
        for (int n = 0; n < TN; ++n)
            Cp[(threadRow * TM + m) * N + threadCol * TN + n] = acc[m * TN + n];
"""


@dataclass(frozen=True)
class MetalKernel:
    kernel: str
    source: str
    multiple: int
    grid: Callable[[int], Dims]
    threadgroup: Dims
    template: tuple[tuple[str, int], ...] = ()
    header: str = ""

    def launch(self, n: int) -> tuple[Dims, Dims]:
        check_size(self.kernel, self.multiple, n)
        return self.grid(n), self.threadgroup


KERNELS = {
    "naive": MetalKernel("naive", NAIVE, 1, lambda n: (n, n, 1), (16, 16, 1)),
    "tile32": MetalKernel(
        "tile32", TILE32, 32, lambda n: (n, n, 1), (32, 32, 1), template=(("T", 32),)
    ),
    "reg4x4": MetalKernel("reg4x4", REG4X4, 64, lambda n: (n // 4, n // 4, 1), (16, 16, 1)),
    "reg8x8": MetalKernel(
        "reg8x8",
        REG8X8,
        128,
        lambda n: (n // 128 * 256, n // 128, 1),
        (256, 1, 1),
        header=REG8X8_HEADER,
    ),
}
