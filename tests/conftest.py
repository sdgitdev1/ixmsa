import os

# The suite tests the kernels, so short calls stay on them; test_short_calls_go_to_sdpa turns the routing on.
os.environ["SAGEATTN_SDPA_SHORT"] = "0"
