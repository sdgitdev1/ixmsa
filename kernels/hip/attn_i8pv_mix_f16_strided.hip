// fp16 Q/O, calls with any head not flagged group, each head on its own path, Q/O by element strides (strided HND, NHD). Body in attn_i8pv.h.
#define I8PV_MIX 1
#define I8PV_BF16 0
#define I8PV_STRIDED 1
#define I8PV_NAME attn_i8pv_mix_f16_strided
#include "attn_i8pv.h"
