// bf16 Q/O, calls with any head not flagged group, each head on its own path, contiguous HND. Body in attn_i8pv.h.
#define I8PV_MIX 1
#define I8PV_BF16 1
#define I8PV_STRIDED 0
#define I8PV_NAME attn_i8pv_mix_bf16
#include "attn_i8pv.h"
