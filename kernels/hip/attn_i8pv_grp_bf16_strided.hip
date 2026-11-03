// bf16 Q/O, heads flagged group, one K scale per 16 keys on interior tiles, Q/O by element strides (strided HND, NHD). Body in attn_i8pv.h.
#define I8PV_GROUP 1
#define I8PV_BF16 1
#define I8PV_STRIDED 1
#define I8PV_NAME attn_i8pv_grp_bf16_strided
#include "attn_i8pv.h"
