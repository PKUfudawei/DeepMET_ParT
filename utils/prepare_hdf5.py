import awkward as ak
import numpy as np
import os, h5py
from coffea.nanoevents import NanoEventsFactory, NanoAODSchema

def pad(x, pad_len, pad_value=0):
    """Pad awkward or numpy array along last axis"""
    if isinstance(x, np.ndarray):
        x = np.pad(x, ((0,0),(0, pad_len - x.shape[1])), mode='constant', constant_values=pad_value)
    elif isinstance(x, ak.Array):
        # safe padding
        x = ak.fill_none(ak.pad_none(x, target=pad_len, axis=-1, clip=True), pad_value)
        # convert to numpy
        x = ak.to_numpy(x, allow_missing=True)
        x = np.nan_to_num(x, nan=pad_value, posinf=pad_value, neginf=pad_value)
    else:
        raise TypeError(f"Unsupported type: {type(x)}")
    return x

def safe_to_numpy(array, default=0.0):
    """Safely convert awkward arrays to numpy arrays"""
    try:
        if isinstance(array, ak.Array):
            # fill None first
            array_filled = ak.fill_none(array, default)
            return ak.to_numpy(array_filled, allow_missing=True)
        else:
            return np.asarray(array)
    except Exception as e:
        print(f"Warning: Failed to convert to numpy, using default: {e}")
        # If fail converting, create default arrays
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
    
    # create HDF5 files
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with h5py.File(output_path, "w") as f:
        pf_ds = None
        event_ds = None
        truth_ds = None
        
        n_events = len(events)
        f.attrs["n_events"] = n_events
        f.attrs["pad_len"] = pad_len
        f.attrs["PF_features"] = np.string_(",".join(PF_features+['PF_px', 'PF_py']))
        f.attrs["event_features"] = np.string_(",".join(event_features))
        f.attrs["event_truths"] = np.string_("px,py")

        # batch-wise processing
        for start in range(0, n_events, batch_size):
            end = min(start + batch_size, n_events)
            batch = events[start:end]
            
            try:
                pf_pt = batch.PF.pt
                # Check all events with PF candidates
                if ak.any(ak.num(pf_pt, axis=1) == 0):
                    print(f"Warning: found no PFcand event in batch {start}-{end}，skipped")
                    continue
                
                # safe sort
                PF_sort_idx = ak.argsort(pf_pt, axis=-1, ascending=False)
                
                # PF features
                pf_array = []
                for feat in PF_features:
                    try:
                        arr = getattr(batch[feat.split('_')[0]], feat.split('_')[1]) if '_' in feat else getattr(batch, feat)
                        arr = arr[PF_sort_idx]
                        arr = pad(arr, pad_len=pad_len, pad_value=pad_value)
                        pf_array.append(arr)
                    except Exception as e:
                        print(f"Warning: processing feature {feat}, {e}")
                        default_arr = np.full((len(batch), pad_len), pad_value, dtype=np.float32)
                        pf_array.append(default_arr)
                
                # compute px, py
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
                        print(f"Error: processing feature {feat}, {e}")
                        default_arr = np.full(len(batch), pad_value, dtype=np.float32)
                        event_array.append(default_arr)
                
                event_array = np.stack(event_array, axis=1)
                event_array = np.nan_to_num(event_array, nan=pad_value, posinf=pad_value, neginf=pad_value)
                
                # Regression target (ground truth)
                try:
                    muon = batch.Muon[batch.Muon.looseId]
                    electron = batch.Electron[batch.Electron.cutBased > 1]
                    
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
                    print(f"Warning: error in computing truth, {e}")
                    truth_array = np.zeros((len(batch), 2), dtype=np.float32)
                
                # Create or expand dataset
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
    dataset_directory = './DeepMET'
    for dirpath, dirnames, filenames in os.walk(dataset_directory):
        for f in filenames:
            if not f.endswith('.root'):
                continue
            file_path = os.path.join(dirpath, f)
            output_dir = dirpath.replace('root', 'hdf5')
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, f.replace('root', 'hdf5'))
            print(f'Processing {file_path} -> {output_path}')
            try:
                preprocess_lazy(file_path, output_path, pad_len=2**12, batch_size=2**10)
                print(f'\tSuccess')
            except Exception as e:
                print(f'\tError: {e}')
                import traceback
                traceback.print_exc()
                continue
