import pandas as pd
from glob import glob
import os, re
import pickle
test_n=pd.read_csv('/home/q359zhan/olinked/data/mouse-scehcd/pglyco-scehcd-mouse-glycan-scored.csv')
# test_n = test_n[test_n['MSGP File Name'].str.contains('rabbit')]
all_s = dict()
for file in glob(os.path.join('/home/q359zhan/olinked/data/mouse-scehcd/mgf/','*.mgf')):

    with open(file) as f:
        print(file)
        for line in f:
            if line.startswith('BEGIN IONS'):
                product_ions_moverz = [line]
            elif line.startswith('TITLE'):
                try:
                    rawfile_pattern = r'File:"(.*?)"'
                    # rawfile = re.split('=|\r|\n|\\\\', line)
                    rawfile = re.search(rawfile_pattern, line)
                    rawfile = rawfile.group(1)
                except:
                    rawfile = file.split('/')[-1].replace('.mgf', '.raw')
                scan_pattern = r'scan=(\d+)'
                # scan = re.split('=|\r|\n', line)
                scan = re.search(scan_pattern, line)
                scan = scan.group(1)
                product_ions_moverz.append(line)
                # print(rawfile + scan)
            else:
                product_ions_moverz.append(line)
                if line.startswith('END IONS'):
                    all_s[rawfile + scan] = product_ions_moverz
with open('/home/q359zhan/olinked/data/mouse-scehcd/mgf/mgf.pkl', 'wb') as f:
    pickle.dump(all_s, f)
# with open('/home/q359zhan/olinked/graphnovo/data/pglyco-mouse/glycan-only/ethcd/mgf.pkl', 'rb') as f:
#     all_s = pickle.load(f)
with open('/home/q359zhan/olinked/data/mouse-scehcd/mgf/mouse-scehcd-test.mgf','w') as f:
    for i, row in test_n.iterrows():
        print(i)
        f.writelines(all_s[row['Spec']])
# test_n=pd.read_csv('/home/q359zhan/olinked/graphnovo/data/pglyco-mouse/glycan-only/ethcd/pglyco-scehcd-mouse-glycan-scored.csv')
# # test_n = test_n[test_n['MSGP File Name'].str.contains('rabbit')]
# with open('/home/q359zhan/olinked/graphnovo/data/pglyco-mouse/glycan-only/ethcd/rabbit.mgf','w') as f:
#     for i, row in test_n.iterrows():
#         print(i)
#         f.writelines(all_s[row['Spec Index']])