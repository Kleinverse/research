"""
gpu_kernel.py
===============
Drop-in replacement for mc_Ehk / lcc_kappas using a fused GPU kernel.

Works on both NVIDIA (CUDA) and AMD (ROCm) via CuPy.
Uses inline Philox4x32-10 -- NO curand_kernel.h or hiprand headers,
so the same source compiles on both backends without modification.

Install CuPy:
    NVIDIA CUDA 12:   pip install cupy-cuda12x
    AMD ROCm 5.x:     pip install cupy-rocm-5-0
    AMD ROCm 6.x:     pip install cupy-rocm-6-0

ROCm device fix for RX 9070 XT (RDNA4 / gfx1201)
--------------------------------------------------
If torch.cuda.is_available() returns False, add to your shell rc:

    export HSA_OVERRIDE_GFX_VERSION=12.0.1
    export ROCR_VISIBLE_DEVICES=0

Then reinstall ROCm PyTorch:
    pip install torch --index-url https://download.pytorch.org/whl/rocm6.2

Usage
-----
    from gpu_kernel import lcc_kappas_native, mc_Ehk_native, warmup
    warmup()
    ehk = mc_Ehk_native(xs_np, k=4, n_mc=500_000)
    lcc = lcc_kappas_native(xs_np, n_mc=500_000)
"""

from __future__ import annotations
import math
import numpy as np

try:
    import cupy as cp
    _CUPY_AVAILABLE = True
except ImportError:
    _CUPY_AVAILABLE = False

_KERNEL_SRC = r"""
typedef unsigned int       u32;
typedef unsigned long long u64;

__device__ __forceinline__ u32 mulhi32(u32 a, u32 b) {
    return (u32)(((u64)a * (u64)b) >> 32);
}
__device__ __forceinline__
void philox_round(u32 c[4], const u32 k[2]) {
    const u32 M0=0xD2511F53u, M1=0xCD9E8D57u;
    u32 lo0=c[0]*M0, hi0=mulhi32(c[0],M0);
    u32 lo1=c[2]*M1, hi1=mulhi32(c[2],M1);
    c[0]=hi1^c[1]^k[0]; c[1]=lo1;
    c[2]=hi0^c[3]^k[1]; c[3]=lo0;
}
__device__ __forceinline__
void philox4x32_10(u32 ctr[4], u32 key[2], u32 out[4]) {
    out[0]=ctr[0]; out[1]=ctr[1]; out[2]=ctr[2]; out[3]=ctr[3];
    u32 k[2]={key[0],key[1]};
    philox_round(out,k); k[0]+=0x9E3779B9u; k[1]+=0xBB67AE85u;
    philox_round(out,k); k[0]+=0x9E3779B9u; k[1]+=0xBB67AE85u;
    philox_round(out,k); k[0]+=0x9E3779B9u; k[1]+=0xBB67AE85u;
    philox_round(out,k); k[0]+=0x9E3779B9u; k[1]+=0xBB67AE85u;
    philox_round(out,k); k[0]+=0x9E3779B9u; k[1]+=0xBB67AE85u;
    philox_round(out,k); k[0]+=0x9E3779B9u; k[1]+=0xBB67AE85u;
    philox_round(out,k); k[0]+=0x9E3779B9u; k[1]+=0xBB67AE85u;
    philox_round(out,k); k[0]+=0x9E3779B9u; k[1]+=0xBB67AE85u;
    philox_round(out,k); k[0]+=0x9E3779B9u; k[1]+=0xBB67AE85u;
    philox_round(out,k);
}
__device__ __forceinline__ int rnd_idx(u32 r, int n) {
    return (int)((u64)r*(u64)n>>32);
}

extern "C" __global__ void vstat_fused(
    const double* __restrict__ xs,
    int n, int n_mc,
    double* out,
    u32 seed_lo, u32 seed_hi)
{
    extern __shared__ double sh[];
    const int tid=blockIdx.x*blockDim.x+threadIdx.x;
    const int lane=threadIdx.x;
    double prod[7];

    if (tid < n_mc) {
        u32 ctr0[4]={(u32)tid,0u,0u,0u}, ctr1[4]={(u32)tid,1u,0u,0u};
        u32 key[2]={seed_lo,seed_hi}, r0[4], r1[4];
        philox4x32_10(ctr0,key,r0);
        philox4x32_10(ctr1,key,r1);
        double z[8];
        z[0]=xs[rnd_idx(r0[0],n)]; z[1]=xs[rnd_idx(r0[1],n)];
        z[2]=xs[rnd_idx(r0[2],n)]; z[3]=xs[rnd_idx(r0[3],n)];
        z[4]=xs[rnd_idx(r1[0],n)]; z[5]=xs[rnd_idx(r1[1],n)];
        z[6]=xs[rnd_idx(r1[2],n)]; z[7]=xs[rnd_idx(r1[3],n)];
        for (int i=0;i<7;i++){
            int k=i+2; double mu=0.0;
            for(int j=0;j<k;j++) mu+=z[j]; mu/=(double)k;
            double p=1.0;
            for(int j=0;j<k;j++) p*=(z[j]-mu);
            prod[i]=p;
        }
    } else {
        for(int i=0;i<7;i++) prod[i]=0.0;
    }

    for(int i=0;i<7;i++) sh[i*blockDim.x+lane]=prod[i];
    __syncthreads();
    for(int s=blockDim.x>>1;s>0;s>>=1){
        if(lane<s) for(int i=0;i<7;i++)
            sh[i*blockDim.x+lane]+=sh[i*blockDim.x+lane+s];
        __syncthreads();
    }
    if(lane==0) for(int i=0;i<7;i++) atomicAdd(&out[i],sh[i*blockDim.x]);
}
"""

_kernel_cache: dict = {}
_BLOCK   = 256
_ORDERS  = [2, 3, 4, 5, 6, 7, 8]
_rng_seq = 0


def _compile_options() -> tuple:
    try:
        cc = str(cp.cuda.Device().compute_capability)
        if len(cc) <= 3:                          # NVIDIA SM string
            return (f"--gpu-architecture=sm_{cc}", "-O3", "--use_fast_math",
                    "--default-stream=per-thread")
        else:                                     # AMD gfx string
            return (f"--amdgpu-target=gfx{cc}",)
    except Exception:
        return ("-O3",)


def _get_kernel():
    if not _CUPY_AVAILABLE:
        raise ImportError(
            "CuPy not found.\n"
            "  NVIDIA:  pip install cupy-cuda12x\n"
            "  AMD:     pip install cupy-rocm-6-0")
    dev_id = cp.cuda.Device().id
    if dev_id not in _kernel_cache:
        _kernel_cache[dev_id] = cp.RawKernel(
            _KERNEL_SRC, "vstat_fused",
            backend="nvcc", options=_compile_options())
    return _kernel_cache[dev_id]


def _seeds():
    global _rng_seq
    _rng_seq += 1
    return (np.uint32(_rng_seq & 0xFFFFFFFF),
            np.uint32((_rng_seq >> 32) & 0xFFFFFFFF))


def _run(xs: np.ndarray, n_mc: int) -> np.ndarray:
    kern    = _get_kernel()
    n       = len(xs)
    xs_gpu  = cp.asarray(xs, dtype=cp.float64)
    out_gpu = cp.zeros(7, dtype=cp.float64)
    grid    = math.ceil(n_mc / _BLOCK)
    lo, hi  = _seeds()
    kern((grid,), (_BLOCK,),
         (xs_gpu, np.int32(n), np.int32(n_mc), out_gpu, lo, hi),
         shared_mem=7 * _BLOCK * 8)
    cp.cuda.Device().synchronize()
    return (out_gpu / n_mc).get()


def mc_Ehk_native(xs: np.ndarray, k: int, n_mc: int) -> float:
    assert k in _ORDERS, f"k={k} not in {_ORDERS}"
    return float(_run(xs, n_mc)[k - 2])


def cyclic_Ehk_native(xs: np.ndarray, k: int) -> float:
    """Exact cyclic E[h_k] over all n cyclic k-tuples (deterministic)."""
    n   = len(xs)
    idx = np.stack([(np.arange(n) + j) % n for j in range(k)], axis=1)
    z   = xs[idx]
    return float((z - z.mean(axis=1, keepdims=True)).prod(axis=1).mean())


def lcc_kappas_native(xs: np.ndarray) -> dict:
    Ehk = {k: cyclic_Ehk_native(xs, k) for k in _ORDERS}

    def alpha(r: int, k: int) -> float:
        return ((-1) ** (r - 1)) * math.factorial(r - 1) / k ** (r - 1)

    def partition_lower(k: int, K: dict) -> float:
        if k <= 3:
            return 0.0
        k2 = K.get(2, 0.0); k3 = K.get(3, 0.0)
        k4 = K.get(4, 0.0); k5 = K.get(5, 0.0); k6 = K.get(6, 0.0)
        if k == 4:
            return 3.0 * alpha(2,4)**2 * k2**2
        if k == 5:
            return 10.0 * alpha(2,5) * alpha(3,5) * k2 * k3
        if k == 6:
            return (15.0 * alpha(2,6) * alpha(4,6) * k2 * k4
                  + 10.0 * alpha(3,6)**2             * k3**2
                  + 15.0 * alpha(2,6)**3              * k2**3)
        if k == 7:
            return (21.0  * alpha(2,7) * alpha(5,7)          * k2 * k5
                  + 35.0  * alpha(3,7) * alpha(4,7)          * k3 * k4
                  + 105.0 * alpha(2,7)**2 * alpha(3,7)       * k2**2 * k3)
        if k == 8:
            return (28.0  * alpha(2,8) * alpha(6,8)          * k2 * k6
                  + 56.0  * alpha(3,8) * alpha(5,8)          * k3 * k5
                  + 35.0  * alpha(4,8)**2                    * k4**2
                  + 210.0 * alpha(2,8)**2 * alpha(4,8)       * k2**2 * k4
                  + 280.0 * alpha(2,8) * alpha(3,8)**2       * k2 * k3**2
                  + 105.0 * alpha(2,8)**4                    * k2**4)
        return 0.0

    K: dict[int, float] = {}
    for k in sorted(_ORDERS):
        K[k] = (Ehk[k] - partition_lower(k, K)) / alpha(k, k)
    return {'K': K}


def warmup() -> None:
    if not _CUPY_AVAILABLE:
        print("  [native] CuPy not available.", flush=True)
        return
    print("  [native] Compiling vstat_fused ...", end=" ", flush=True)
    _run(np.random.randn(10).astype(np.float64), n_mc=256)
    cc = cp.cuda.Device().compute_capability
    print(f"done  (SM/GFX {cc}  block={_BLOCK})", flush=True)
