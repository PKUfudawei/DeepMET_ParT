import awkward as ak
import numpy as np
import os, h5py
from coffea.nanoevents import NanoEventsFactory, NanoAODSchema

def pad(x, pad_len, pad_value=0):
    """Pad awkward or numpy array along last axis"""
    if isinstance(x, np.ndarray):
        x = np.pad(x, ((0,0),(0, pad_len - x.shape[1])), mode='constant', constant_values=pad_value)
    elif isinstance(x, ak.Array):
        # 使用更安全的padding方法
        x = ak.fill_none(ak.pad_none(x, target=pad_len, axis=-1, clip=True), pad_value)
        # 直接转换为numpy，避免后续的ak.to_numpy问题
        x = ak.to_numpy(x, allow_missing=True)
        x = np.nan_to_num(x, nan=pad_value, posinf=pad_value, neginf=pad_value)
    else:
        raise TypeError(f"Unsupported type: {type(x)}")
    return x

def safe_to_numpy(array, default=0.0):
    """安全地将awkward数组转换为numpy数组"""
    try:
        if isinstance(array, ak.Array):
            # 先填充None值
            array_filled = ak.fill_none(array, default)
            # 尝试转换为numpy
            return ak.to_numpy(array_filled, allow_missing=True)
        else:
            return np.asarray(array)
    except Exception as e:
        print(f"Warning: Failed to convert to numpy, using default: {e}")
        # 如果转换失败，创建默认数组
        if hasattr(array, '__len__'):
            return np.full(len(array), default, dtype=np.float32)
        else:
            return np.array([default], dtype=np.float32)

def preprocess_lazy(file_path, output_path, pad_len=4500, pad_value=0, batch_size=2**10):  # 减小batch_size
    events = NanoEventsFactory.from_root(
        {file_path: 'Events'}, schemaclass=NanoAODSchema, mode="virtual"
    ).events()
    
    PF_features = [
        'PF_pt', 'PF_eta', 'PF_phi', 'PF_mass', 'PF_d0', 'PF_dz', 'PF_hcalFraction', 
        'PF_pdgId', 'PF_charge', 'PF_fromPV', 'PF_puppiWeightNoLep', 'PF_puppiWeight'
    ]
    event_features = ['fixedGridRhoFastjetAll', 'fixedGridRhoFastjetCentralCalo', 'PV_npvs', 'PV_npvsGood']
    
    # 创建 HDF5 文件
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with h5py.File(output_path, "w") as f:
        pf_ds = None
        event_ds = None
        truth_ds = None
        
        # 分批处理
        n_events = len(events)
        f.attrs["n_events"] = n_events
        f.attrs["pad_len"] = pad_len
        f.attrs["PF_features"] = np.string_(",".join(PF_features+['PF_px', 'PF_py']))
        f.attrs["event_features"] = np.string_(",".join(event_features))
        f.attrs["event_truths"] = np.string_("px,py")

        for start in range(0, n_events, batch_size):
            end = min(start + batch_size, n_events)
            batch = events[start:end]
            
            try:
                # 安全的PF排序
                pf_pt = batch.PF.pt
                # 检查是否有PF粒子
                if ak.num(pf_pt, axis=1)[0] == 0:
                    print(f"Warning: 批次 {start}-{end} 没有PF粒子，跳过")
                    continue
                
                # 使用更安全的排序方法
                PF_sort_idx = ak.argsort(pf_pt, axis=-1, ascending=False)
                
                # PF features - 使用安全的方法处理
                pf_array = []
                for feat in PF_features:
                    try:
                        arr = getattr(batch[feat.split('_')[0]], feat.split('_')[1]) if '_' in feat else getattr(batch, feat)
                        arr = arr[PF_sort_idx]
                        arr = pad(arr, pad_len=pad_len, pad_value=pad_value)
                        pf_array.append(arr)
                    except Exception as e:
                        print(f"Warning: 处理特征 {feat} 时出错: {e}")
                        # 创建默认数组
                        default_arr = np.full((len(batch), pad_len), pad_value, dtype=np.float32)
                        pf_array.append(default_arr)
                
                # 计算px, py
                pf_pt_arr = pf_array[0]  # PF_pt
                pf_phi_arr = pf_array[2]  # PF_phi
                
                pf_px = pf_pt_arr * np.cos(pf_phi_arr)
                pf_py = pf_pt_arr * np.sin(pf_phi_arr)
                
                pf_array.append(pf_px)
                pf_array.append(pf_py)
                
                pf_array = np.stack(pf_array, axis=1)
                pf_array = np.nan_to_num(pf_array, nan=pad_value, posinf=pad_value, neginf=pad_value)
                
                # event features
                event_array = []
                for feat in event_features:
                    try:
                        arr = getattr(batch[feat.split('_')[0]], feat.split('_')[1]) if '_' in feat else getattr(batch, feat)
                        arr = safe_to_numpy(arr, pad_value)
                        event_array.append(arr)
                    except Exception as e:
                        print(f"Warning: 处理事件特征 {feat} 时出错: {e}")
                        default_arr = np.full(len(batch), pad_value, dtype=np.float32)
                        event_array.append(default_arr)
                
                event_array = np.stack(event_array, axis=1)
                event_array = np.nan_to_num(event_array, nan=pad_value, posinf=pad_value, neginf=pad_value)
                
                # target - 使用安全的方法
                try:
                    muon = batch.Muon[batch.Muon.looseId]
                    electron = batch.Electron[batch.Electron.cutBased > 1]
                    
                    # 安全的求和
                    muon_px = safe_to_numpy(ak.sum(muon.pt * np.cos(muon.phi), axis=-1), 0.0)
                    muon_py = safe_to_numpy(ak.sum(muon.pt * np.sin(muon.phi), axis=-1), 0.0)
                    
                    electron_px = safe_to_numpy(ak.sum(electron.pt * np.cos(electron.phi), axis=-1), 0.0)
                    electron_py = safe_to_numpy(ak.sum(electron.pt * np.sin(electron.phi), axis=-1), 0.0)
                    
                    genmet_px = safe_to_numpy(batch.GenMET.pt * np.cos(batch.GenMET.phi), 0.0)
                    genmet_py = safe_to_numpy(batch.GenMET.pt * np.sin(batch.GenMET.phi), 0.0)
                    
                    truth_px = genmet_px + muon_px
                    truth_py = genmet_py + muon_py
                    
                    if 'MuMu' not in file_path:
                        truth_px += electron_px
                        truth_py += electron_py
                        
                    truth_array = np.stack([truth_px, truth_py], axis=1)
                    truth_array = np.nan_to_num(truth_array, nan=0.0, posinf=1e6, neginf=-1e6)
                    
                except Exception as e:
                    print(f"Warning: 计算truth时出错: {e}")
                    truth_array = np.zeros((len(batch), 2), dtype=np.float32)
                
                # 创建或扩展dataset
                if pf_ds is None:
                    pf_ds = f.create_dataset("PF_features", data=pf_array, maxshape=(None, pf_array.shape[1], pf_array.shape[2]), compression="lzf")
                    event_ds = f.create_dataset("event_features", data=event_array, maxshape=(None, event_array.shape[1]), compression="lzf")
                    truth_ds = f.create_dataset("event_truths", data=truth_array, maxshape=(None, truth_array.shape[1]), compression="lzf")
                else:
                    pf_ds.resize(pf_ds.shape[0] + pf_array.shape[0], axis=0)
                    pf_ds[-pf_array.shape[0]:] = pf_array
                    event_ds.resize(event_ds.shape[0] + event_array.shape[0], axis=0)
                    event_ds[-event_array.shape[0]:] = event_array
                    truth_ds.resize(truth_ds.shape[0] + truth_array.shape[0], axis=0)
                    truth_ds[-truth_array.shape[0]:] = truth_array
                    
            except Exception as e:
                print(f"Error processing batch {start}-{end}: {e}")
                continue

if __name__ == "__main__":
    top_directory = './DeepMET'
    for dirpath, dirnames, filenames in os.walk(top_directory):
        for f in filenames:
            if not f.endswith('.root'):
                continue
            file_path = os.path.join(dirpath, f)
            output_path = os.path.join(dirpath, f.replace('.root', '.hdf5')).replace('DeepMET', 'DeepMET_hdf5')
            print(f'Processing {file_path} -> {output_path}')
            try:
                preprocess_lazy(file_path, output_path, pad_len=4500, batch_size=2**10)  # 使用更小的batch_size
                print(f'Successfully processed {file_path}')
            except Exception as e:
                print(f'Error processing {file_path}: {e}')
                import traceback
                traceback.print_exc()
                continue