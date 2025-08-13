### Install pointnet2
cd Pose_Estimation_Model/model/pointnet2
pip install --user .
cd ../../../

### Download ISM pretrained model
cd Instance_Segmentation_Model
python3 download_sam.py
python3 download_fastsam.py
python3 download_dinov2.py
cd ../

### Download PEM pretrained model
cd Pose_Estimation_Model
python3 download_sam6d-pem.py
