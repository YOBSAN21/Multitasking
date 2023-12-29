import os
import numpy as np
import shutil

# rootdir = "F:/willie/OPENSARSHIP_2/Test"
#
# classes = ['Tanker'] #, 'Cargo_vh']   #, 'Fishing', 'Others', 'Passenger', 'Tanker', 'Tug']
#
# for i in classes:
#
#     os.makedirs(rootdir + '/Test/' + i)
#     os.makedirs(rootdir + '/Val/' + i)
#
#     source = rootdir + '/' + i
#
#     allFileNames = os.listdir(source)
#     np.random.shuffle(allFileNames)
#     # test_ratio = 0.70
#     val_ratio = 0.166
#
#     # train_FileNames, test_FileNames = np.split(np.array(allFileNames), [int(len(allFileNames)* (1 - test_ratio))])
#     test_FileNames, val_FileNames = np.split(np.array(allFileNames), [int(len(allFileNames) * (1 - val_ratio))])
#
#     # train_FileNames = [source+'/'+ name for name in train_FileNames.tolist()]
#     test_FileNames = [source+'/' + name for name in test_FileNames.tolist()]
#     val_FileNames = [source + '/' + name for name in val_FileNames.tolist()]
#
#     # for name in train_FileNames:
#     #     shutil.copy(name, rootdir +'/Train/' + i)
#
#     for name in test_FileNames:
#         shutil.copy(name, rootdir +'/Test/' + i)
#
#     for name in val_FileNames:
#         shutil.copy(name, rootdir +'/Val/' + i)


# rootdir= "C:/Users/PC/Desktop/Multi/Data"
rootdir= "C:/Users/PC/Desktop/Binary"


classes = ['Stroke']  # ['Bones', 'Brain', 'Normal']

for i in classes:

    os.makedirs(rootdir + '/Test/' + i)
    os.makedirs(rootdir + '/Val/' + i)

    source = rootdir + '/' + i

    allFileNames = os.listdir(source)
    np.random.shuffle(allFileNames)
    # test_ratio = 0.075
    val_ratio = 0.56

    # train_FileNames, test_FileNames = np.split(np.array(allFileNames), [int(len(allFileNames)* (1 - test_ratio))])
    test_FileNames, val_FileNames = np.split(np.array(allFileNames), [int(len(allFileNames) * (1 - val_ratio))])

    # train_FileNames = [source+'/'+ name for name in train_FileNames.tolist()]
    test_FileNames = [source+'/' + name for name in test_FileNames.tolist()]
    val_FileNames = [source + '/' + name for name in val_FileNames.tolist()]

    # for name in train_FileNames:
    #     shutil.copy(name, rootdir +'/Train/' + i)

    for name in test_FileNames:
        shutil.copy(name, rootdir +'/Test/' + i)

    for name in val_FileNames:
        shutil.copy(name, rootdir +'/Val/' + i)
