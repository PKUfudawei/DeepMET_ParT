import awkward as ak
import numpy as np
import os

from coffea.nanoevents import NanoEventsFactory, NanoAODSchema


def preprocess(file_path, output_path):
    # read NanoAOD root file
    events = NanoEventsFactory.from_root(
        {file_path: 'Events'}, schemaclass=NanoAODSchema, mode="eager"
    ).events()
    
    # input features
    variable = {'numMuon': ak.num(events.Muon)}

    for i in [
        'PF_pt', 'PF_eta', 'PF_phi', 'PF_mass', 'PF_d0', 'PF_dz', 
        'PF_hcalFraction', 'PF_pdgId', 'PF_charge', 'PF_fromPV',
        'PF_puppiWeightNoLep', 'PF_puppiWeight', 
        'fixedGridRhoFastjetAll', 'fixedGridRhoFastjetCentralCalo', 'PV_npvs', 'PV_npvsGood', 
    ]:
        if '_' in i:
            variable[i] = getattr(events[i.split('_')[0]], i.split('_')[1])
        else:
            variable[i] = getattr(events, i)

    variable['PF_px'] = variable['PF_pt'] * np.cos(variable['PF_phi'])
    variable['PF_py'] = variable['PF_pt'] * np.sin(variable['PF_phi'])
    

    # ground truth information
    variable['muons_px'] = ak.sum(events.Muon.pt * np.cos(events.Muon.phi), axis=1)
    variable['muons_py'] = ak.sum(events.Muon.pt * np.sin(events.Muon.phi), axis=1)
    variable['truth_px'] = variable['muons_px']
    variable['truth_py'] = variable['muons_py']
    if 'GenMET' in events.fields:
        variable['GenMET_px'] = events.GenMET.pt * np.cos(events.GenMET.phi)
        variable['GenMET_py'] = events.GenMET.pt * np.sin(events.GenMET.phi)
        variable['truth_px'] = variable['truth_px'] + variable['GenMET_px']
        variable['truth_py'] = variable['truth_py'] + variable['GenMET_py']
    variable['truth_pt'] = np.sqrt(variable['truth_px']**2 + variable['truth_py']**2)
    variable['truth_phi'] = np.arctan2(variable['truth_py'], variable['truth_px'])

    # store the variables in a Parquet file
    ak.to_parquet(array=ak.Array(variable), destination=output_path, compression='zstd')


if __name__ == "__main__":
    top_directory = '/ospool/cms-user/fudawei/DeepMETTraining'

    for dirpath, dirnames, filenames in os.walk(top_directory):
        if len(filenames) == 0:
            continue
        for f in filenames:
            if f.endswith('.root'):
                file_path = os.path.join(dirpath, f)
                output_path = os.path.join(dirpath, f.replace('.root', '.parquet'))
                print(f'Processing {file_path} -> {output_path}')
                preprocess(file_path, output_path)
