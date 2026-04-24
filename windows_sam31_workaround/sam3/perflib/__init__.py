# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

# pyre-unsafe

import os

is_enabled = False
default_perflib = "0" if os.name == "nt" else "1"
if os.getenv("USE_PERFLIB", default_perflib) == "1":
    # print("Enabled the use of perflib.\n", end="")
    is_enabled = True
