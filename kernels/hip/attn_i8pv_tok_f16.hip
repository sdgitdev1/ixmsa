// fp16 Q/O, the other heads, per-token K scales, contiguous HND. Body in attn_i8pv.h.
#define I8PV_GROUP 0
#define I8PV_BF16 0
#define I8PV_STRIDED 0
#define I8PV_NAME attn_i8pv_tok_f16
#include "attn_i8pv.h"
