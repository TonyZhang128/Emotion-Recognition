import torch

CKPT = r"D:\song_LD\save_model\multi_try\B_lr0.0001_wd0.0003_dp0.5\best_model_fold1.pth"  # 改成你的

obj = torch.load(CKPT, map_location="cpu")

print("=== type(obj) =", type(obj))
if isinstance(obj, dict):
    print("=== top-level keys ===")
    for k in obj.keys():
        print("  -", k)

    # 常见的 state_dict key 候选
    cand = ["state_dict", "model_state", "model", "net", "teacher", "student"]
    found = None
    for k in cand:
        if k in obj and isinstance(obj[k], dict):
            found = k
            break

    if found is not None:
        sd = obj[found]
        print(f"\n[OK] state_dict seems to be under key: '{found}'")
    else:
        # 也可能顶层就是 state_dict（直接是参数名->tensor）
        # 判断方式：看看 key 像不像 'xxx.weight'
        keys = list(obj.keys())
        looks_like_sd = any(isinstance(keys[i], str) and (".weight" in keys[i] or ".bias" in keys[i])
                            for i in range(min(20, len(keys))))
        if looks_like_sd:
            sd = obj
            print("\n[OK] top-level dict itself looks like a state_dict.")
        else:
            sd = None
            print("\n[WARN] cannot find state_dict automatically. Show some values types:")
            for k in keys[:10]:
                print(k, type(obj[k]))

    if sd is not None:
        print("\n=== first 30 parameter names in state_dict ===")
        for i, name in enumerate(list(sd.keys())[:30]):
            print(f"{i:02d} {name}  {tuple(sd[name].shape) if hasattr(sd[name], 'shape') else type(sd[name])}")

else:
    print("[WARN] ckpt is not a dict; it might already be a state_dict-like object.")
