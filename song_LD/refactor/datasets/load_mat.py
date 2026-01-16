import scipy.io as sio
import numpy as np

def load_mat_var(path: str, candidates: List[str], squeeze: bool = True) -> np.ndarray:
    """
    通用MAT文件读取函数,支持v7.3/cell/ASCII格式

    Args:
        path: MAT文件路径
        candidates: 候选变量名列表
        squeeze: 是否压缩维度

    Returns:
        numpy数组
    """
    arr = None
    need_h5py = False

    # 尝试使用scipy.io读取
    try:
        m = sio.loadmat(path)
        for k in candidates:
            if k in m:
                arr = m[k]
                break
        if arr is None:
            for k, v in m.items():
                if not k.startswith('__'):
                    arr = v
                    break
    except NotImplementedError:
        need_h5py = True
    except Exception as e:
        if any(s in str(e).lower() for s in ["v7.3", "hdf"]):
            need_h5py = True
        else:
            raise

    # 使用h5py读取v7.3格式
    if arr is None and need_h5py:
        import h5py

        def _deref_any(obj, f):
            """递归解析h5py对象"""
            if isinstance(obj, (h5py.Reference, h5py.h5r.Reference)):
                if not obj:
                    return None
                return _deref_any(f[obj], f)

            if isinstance(obj, h5py.Group):
                for name in obj.keys():
                    got = _deref_any(obj[name], f)
                    if got is not None:
                        return got
                return None

            if isinstance(obj, h5py.Dataset):
                data = obj[()]
                if isinstance(data, np.ndarray) and data.dtype.kind == 'O':
                    flat = []
                    for el in data.flat:
                        if isinstance(el, (h5py.Reference, h5py.h5r.Reference)):
                            val = _deref_any(el, f)
                        else:
                            val = el

                        if isinstance(val, np.ndarray):
                            if val.dtype.kind in ('U', 'S') or getattr(val.dtype, "char", '') == 'S':
                                try:
                                    s = ''.join(val.reshape(-1).astype(str).tolist())
                                except Exception:
                                    s = str(val)
                                flat.append(s)
                            elif np.issubdtype(val.dtype, np.integer) and val.size <= 16:
                                try:
                                    vv = val.reshape(-1)
                                    if np.all((vv >= 32) & (vv <= 126)):
                                        s = ''.join(chr(int(c)) for c in vv)
                                        flat.append(s)
                                    else:
                                        flat.append(vv[0] if vv.size >= 1 else np.nan)
                                except Exception:
                                    vv = val.reshape(-1)
                                    flat.append(vv[0] if vv.size >= 1 else np.nan)
                            else:
                                vv = val.reshape(-1)
                                flat.append(vv[0] if vv.size >= 1 else np.nan)
                        else:
                            flat.append(val)

                    out = np.array(flat, dtype=object).reshape(data.shape)
                    return out
                else:
                    return data
            return None

        with h5py.File(path, 'r') as f:
            for k in candidates:
                if k in f:
                    arr = _deref_any(f[k], f)
                    break
            if arr is None:
                for name in f.keys():
                    arr = _deref_any(f[name], f)
                    if arr is not None:
                        break
            if arr is None:
                raise KeyError(f"HDF5中未找到候选变量: {candidates}; 顶层键={list(f.keys())}")

    if arr is None:
        raise KeyError(f"MAT文件中未找到候选变量: {candidates}")

    arr = np.asarray(arr)
    if squeeze:
        arr = np.squeeze(arr)
    return arr

