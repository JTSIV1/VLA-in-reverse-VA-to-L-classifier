"""Quick inspection of DROID RLDS feature keys from a local shard."""
import tensorflow as tf
import numpy as np

shard = "/data/user_data/wenjiel2/datasets/droid_rlds/droid_101-train.tfrecord-00000-of-02048"
ds = tf.data.TFRecordDataset(shard)

for raw in ds:
    ex = tf.train.Example()
    ex.ParseFromString(raw.numpy())
    feat = ex.features.feature

    print("=== All feature keys ===")
    for k in sorted(feat.keys()):
        f = feat[k]
        if f.bytes_list.value:
            n = len(f.bytes_list.value)
            sz = len(f.bytes_list.value[0]) if n > 0 else 0
            # Check if first entry looks like text
            try:
                sample = f.bytes_list.value[0].decode()[:80]
                print(f"  {k}: bytes_list[{n}], first_len={sz}, text='{sample}'")
            except Exception:
                print(f"  {k}: bytes_list[{n}], first_len={sz} (binary)")
        elif f.int64_list.value:
            vals = list(f.int64_list.value)
            print(f"  {k}: int64[{len(vals)}] = {vals[:5]}{'...' if len(vals)>5 else ''}")
        elif f.float_list.value:
            vals = list(f.float_list.value)
            print(f"  {k}: float[{len(vals)}] first5={vals[:5]}{'...' if len(vals)>5 else ''}")
        else:
            print(f"  {k}: empty")
    break
