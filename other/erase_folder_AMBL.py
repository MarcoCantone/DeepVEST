import os
import shutil

dataset_path = "C:\\Users\\marco\\Desktop\\Advanced-MRI-Breast-Lesions-Copia\\Advanced-MRI-Breast-Lesions"
clients = os.listdir(dataset_path)

monai_data = []

for client in clients:
    if not os.path.isdir(os.path.join(dataset_path, client)):
        continue
    study = os.listdir(os.path.join(dataset_path, client))[0]
    sequences = os.listdir(os.path.join(dataset_path, client, study))

    img = [x for x in sequences if "-AX Sen Vibrant MultiPhase" in x][0]
    registered = [x for x in sequences if "-Registered AX Sen Vibrant MultiPhase" in x][0]
    seg = [x for x in sequences if "-ROI" in x][0]

    for seq in set(sequences).difference({img, registered, seg}):
        print(os.path.join(dataset_path, client, study, seq))
        shutil.rmtree(os.path.join(dataset_path, client, study, seq))

