// fp16 Q/O, calls whose every head is flagged group, one K scale per 16 keys on interior tiles, contiguous HND. Body in attn_i8pv.h.
#define I8PV_MIX 0
#define I8PV_BF16 0
#define I8PV_STRIDED 0
#define I8PV_NAME attn_i8pv_grp_f16
#include "attn_i8pv.h"
