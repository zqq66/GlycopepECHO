import pickle
import numpy as np
from glob import glob
import re
import os
import sys
from BasicClass import Composition
diag_ion=[204.086646, 186.076086, 168.065526, 366.139466, 144.0656, 138.055, 512.197375, 292.1026925, 274.0921325,
          657.2349, 243.026426, 405.079246, 485.045576, 308.09761]
diag_ion = np.array(diag_ion)
raw_mgf_blocks = {}
all_pre={}
file_path= sys.argv[1]
scans = []
for file in glob(os.path.join(file_path,'*.mgf')):

    with open(file) as f:
        for line in f:
            if line.startswith('BEGIN IONS'):
                product_ions_moverz = []
                product_ions_intensity = []
            elif line.startswith('PEPMASS'):
                mz = float(re.split('=|\r|\n|\s', line)[1])
            elif line.startswith('CHARGE'):
                z = int(re.search(r'CHARGE=(\d+)\+', line).group(1))
            elif line.startswith('TITLE'):
                # rawfile_pattern = r'File:"(.*?)"'
                # # rawfile = re.split('=|\r|\n|\\\\', line)
                # rawfile = re.search(rawfile_pattern, line)
                # rawfile = rawfile.group(1)
                scan_pattern = r'scan=(\d+)'
                # scan = re.split('=|\r|\n', line)
                scan = re.search(scan_pattern, line)
                scan = scan.group(1)
                
            elif line[0].isnumeric():
                product_ion_moverz, product_ion_intensity = line.strip().split(' ')
                product_ions_moverz.append(float(product_ion_moverz))
                product_ions_intensity.append(int(float(product_ion_intensity)))
            elif line.startswith('END IONS'):
                moverzs =np.array(product_ions_moverz)
                start_index = moverzs.searchsorted(diag_ion - 0.05)
                end_index = moverzs.searchsorted(diag_ion + 0.05)
                diag_int = np.concatenate([product_ions_intensity[start:end] for start, end in zip(start_index, end_index)])
                # print(diag_int, diag_int>1e5, 1e5)
                if np.any(diag_int>1e5):# np.any(end_index > start_index): #
                    rawfile = file.split('.')[0].split('/')[-1] + '.raw'
                    raw_mgf_blocks[rawfile + scan] = {'product_ions_moverz': np.array(product_ions_moverz),
                                                    'product_ions_intensity': np.array(product_ions_intensity)}
                    all_pre[rawfile + scan] = {'precursor_mass': mz * z - Composition('proton').mass * (z - 1),
                                            'precursor_charge': z}
                    print(rawfile + scan)
                    scans.append(int(scan))
print(len(raw_mgf_blocks.keys()))

with open(os.path.join(file_path, 'high_filtered_mgf.pkl'), 'wb') as f:
    pickle.dump(raw_mgf_blocks, f)

with open(os.path.join(file_path, 'high_filtered_precursor.pkl'), 'wb') as f:
    pickle.dump(all_pre, f)
