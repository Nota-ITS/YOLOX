from mlops_utils.data import kitti_2_voc, get_data_from_nas

data_dir = "datasets"
get_data_from_nas(input_csv="./csv_export.csv", data_dir=data_dir)
kitti_2_voc(data_dir=data_dir)