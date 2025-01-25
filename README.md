# Different RMSNorm implementations for AMD MI300X

Optional dependencies include liger and apex.
Please install them by following their respective READMEs.

The repo also provides a standalone HIP implementation based on Apex kernels - please run the `compile_apex.py` script to compile it.
The compilation process uses PyTorch's extension mechanism to compile the source code, therefore please make sure a correct PyTorch installation is present.

Afterwards `test.py` can be used to test correctness of different implementations.
Example output should look like the following:

```
Shape            Type            Impl.                Fwd.    Bwd.
---------------  --------------  -------------------  ------  ------
(1, 4096, 8192)  torch.float32   apex                 OK      OK
(1, 4096, 8192)  torch.float32   custom-apex          OK      OK
(1, 4096, 8192)  torch.float32   liger                OK      OK
(1, 4096, 8192)  torch.float32   liger-inplace        OK      OK
(1, 4096, 8192)  torch.float32   orig-llama           OK      OK
(1, 4096, 8192)  torch.float32   orig-llama-compiled  OK      OK
(1, 4096, 8192)  torch.float32   pytorch-compiled     OK      OK

(1, 4096, 8192)  torch.float16   apex                 OK      OK
(1, 4096, 8192)  torch.float16   custom-apex          OK      OK
(1, 4096, 8192)  torch.float16   liger                OK      OK
(1, 4096, 8192)  torch.float16   liger-inplace        OK      OK
(1, 4096, 8192)  torch.float16   orig-llama           OK      OK
(1, 4096, 8192)  torch.float16   orig-llama-compiled  OK      OK
(1, 4096, 8192)  torch.float16   pytorch-compiled     OK      OK

(1, 4096, 8192)  torch.bfloat16  apex                 OK      OK
(1, 4096, 8192)  torch.bfloat16  custom-apex          OK      OK
(1, 4096, 8192)  torch.bfloat16  liger                OK      OK
(1, 4096, 8192)  torch.bfloat16  liger-inplace        OK      OK
(1, 4096, 8192)  torch.bfloat16  orig-llama           OK      OK
(1, 4096, 8192)  torch.bfloat16  orig-llama-compiled  OK      OK
(1, 4096, 8192)  torch.bfloat16  pytorch-compiled     OK      OK
```

If functional tests are all ok, benchmarking can then be performed by running `benchmark.py`.

```
Shape            Type            Impl.                Inf.             Fwd.             Bwd.
---------------  --------------  -------------------  ---------------  ---------------  ---------------
(1, 4096, 8192)  torch.float32   apex                 0.1189 ± 0.0142  0.0982 ± 0.0017  0.4221 ± 0.0025
(1, 4096, 8192)  torch.float32   custom-apex          0.1092 ± 0.0020  0.0996 ± 0.0016  0.4237 ± 0.0023
(1, 4096, 8192)  torch.float32   liger                0.0974 ± 0.0153  0.0902 ± 0.0170  0.2276 ± 0.0032
(1, 4096, 8192)  torch.float32   liger-inplace        0.0934 ± 0.0116  0.0828 ± 0.0117  0.2044 ± 0.0030
(1, 4096, 8192)  torch.float32   orig-llama           0.2795 ± 0.0147  0.2774 ± 0.0123  0.4333 ± 0.0023
(1, 4096, 8192)  torch.float32   orig-llama-compiled  0.1053 ± 0.0150  0.0863 ± 0.0097  0.1552 ± 0.0024
(1, 4096, 8192)  torch.float32   pytorch              0.2848 ± 0.0118  0.2899 ± 0.0115  0.4264 ± 0.0046
(1, 4096, 8192)  torch.float32   pytorch-compiled     0.0982 ± 0.0141  0.0859 ± 0.0090  0.1557 ± 0.0047

(1, 4096, 8192)  torch.float16   apex                 0.0502 ± 0.0012  0.0506 ± 0.0008  0.2028 ± 0.0015
(1, 4096, 8192)  torch.float16   custom-apex          0.0558 ± 0.0096  0.0559 ± 0.0059  0.2037 ± 0.0042
(1, 4096, 8192)  torch.float16   liger                0.0784 ± 0.0088  0.0664 ± 0.0154  0.1432 ± 0.0067
(1, 4096, 8192)  torch.float16   liger-inplace        0.0806 ± 0.0043  0.0771 ± 0.0023  0.1266 ± 0.0012
(1, 4096, 8192)  torch.float16   orig-llama           0.3894 ± 0.0177  0.3900 ± 0.0140  0.3434 ± 0.0071
(1, 4096, 8192)  torch.float16   orig-llama-compiled  0.0519 ± 0.0019  0.1104 ± 0.0356  0.1137 ± 0.0030
(1, 4096, 8192)  torch.float16   pytorch              0.1873 ± 0.0136  0.1829 ± 0.0093  0.3446 ± 0.0051
(1, 4096, 8192)  torch.float16   pytorch-compiled     0.0548 ± 0.0065  0.0595 ± 0.0054  0.1143 ± 0.0043

(1, 4096, 8192)  torch.bfloat16  apex                 0.0590 ± 0.0008  0.0600 ± 0.0011  0.2075 ± 0.0048
(1, 4096, 8192)  torch.bfloat16  custom-apex          0.0708 ± 0.0099  0.0674 ± 0.0101  0.2080 ± 0.0051
(1, 4096, 8192)  torch.bfloat16  liger                0.0598 ± 0.0083  0.0563 ± 0.0041  0.1641 ± 0.0029
(1, 4096, 8192)  torch.bfloat16  liger-inplace        0.0584 ± 0.0043  0.0574 ± 0.0051  0.1486 ± 0.0009
(1, 4096, 8192)  torch.bfloat16  orig-llama           0.3952 ± 0.0184  0.3936 ± 0.0165  0.3516 ± 0.0057
(1, 4096, 8192)  torch.bfloat16  orig-llama-compiled  0.0555 ± 0.0067  0.0626 ± 0.0254  0.1137 ± 0.0024
(1, 4096, 8192)  torch.bfloat16  pytorch              0.1882 ± 0.0046  0.1854 ± 0.0044  0.3524 ± 0.0053
(1, 4096, 8192)  torch.bfloat16  pytorch-compiled     0.0546 ± 0.0067  0.0599 ± 0.0043  0.1142 ± 0.0024
```

When benchmarking, `Inf.` refers to performing a forward pass within `torch.inference_mode()` context, while `Fwd.` measures forward time for the purpose of the following backward.
