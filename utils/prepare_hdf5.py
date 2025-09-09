import awkward as ak
import numpy as np
import os, h5py
from coffea.nanoevents import NanoEventsFactory, NanoAODSchema

def pad(x, pad_len, pad_value=0):
    """Pad awkward or numpy array along last axis"""
    if isinstance(x, np.ndarray):
        x = np.pad(x, ((0,0),(0, pad_len - x.shape[1])), mode='constant', constant_values=pad_value)
    elif isinstance(x, ak.Array):
        x = ak.fill_none(ak.pad_none(x, target=pad_len, axis=-1, clip=True), pad_value)
    else:
        raise TypeError(f"Unsupported type: {type(x)}")
    return x

def preprocess_lazy(file_path, output_path, pad_len=4500, pad_value=0, batch_size=2**12):
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
            
            # 排序 PF
            PF_sort_idx = ak.argsort(batch.PF.pt, axis=-1, ascending=False)
            
            # PF features
            pf_array = []
            for feat in PF_features:
                arr = getattr(batch[feat.split('_')[0]], feat.split('_')[1]) if '_' in feat else getattr(batch, feat)
                arr = arr[PF_sort_idx]
                arr = pad(arr, pad_len=pad_len, pad_value=pad_value)
                pf_array.append(ak.to_numpy(arr))
            pf_array.append(pf_array[0] * np.cos(pf_array[2]))
            pf_array.append(pf_array[0] * np.sin(pf_array[2]))
            pf_array = np.stack(pf_array, axis=1)  # shape [batch, n_features, n_PF]
            
            # event features
            event_array = []
            for feat in event_features:
                arr = getattr(batch[feat.split('_')[0]], feat.split('_')[1]) if '_' in feat else getattr(batch, feat)
                event_array.append(ak.to_numpy(arr))
            event_array = np.stack(event_array, axis=1)  # shape [batch, n_event_features]
            
            # target
            muon = batch.Muon[batch.Muon.looseId]
            muon = muon[ak.argsort(muon.pt, axis=-1, ascending=False)]
            electron = batch.Electron[batch.Electron.cutBased>1]
            electron = electron[ak.argsort(electron.pt, axis=-1, ascending=False)]
            
            truth_px = batch.GenMET.pt * np.cos(batch.GenMET.phi) + ak.sum(muon.pt * np.cos(muon.phi), axis=-1)
            truth_py = batch.GenMET.pt * np.sin(batch.GenMET.phi) + ak.sum(muon.pt * np.sin(muon.phi), axis=-1)
            if 'MuMu' not in file_path:
                truth_px = truth_px + ak.sum(electron.pt * np.cos(electron.phi), axis=-1)
                truth_py = truth_py + ak.sum(electron.pt * np.sin(electron.phi), axis=-1)
            
            truth_array = np.stack([ak.to_numpy(truth_px), ak.to_numpy(truth_py)], axis=1)
            
            # 创建 dataset 或 append
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

if __name__ == "__main__":
    top_directory = './DeepMET'
    for dirpath, dirnames, filenames in os.walk(top_directory):
        for f in filenames:
            if not f.endswith('.root'):
                continue
            file_path = os.path.join(dirpath, f)
            output_path = os.path.join(dirpath, f.replace('.root', '.hdf5')).replace('DeepMET', 'DeepMET_hdf5')
            print(f'Processing {file_path} -> {output_path}')
            preprocess_lazy(file_path, output_path, pad_len=4500, batch_size=2**12)
