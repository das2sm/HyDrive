# cp -r /horizon-bucket/HCT_Perception2/users/wenchao01.sun/SparseDrive/ckpt/ .
# cp -r /horizon-bucket/HCT_Perception2/users/wenchao01.sun/SparseDrive/data/ .

# mkdir data/nuscenes
# ln -s /horizon-bucket/HCT_Perception2/public_dataset/vad_nuscenes/maps data/nuscenes/maps
# ln -s /horizon-bucket/HCT_Perception2/public_dataset/vad_nuscenes/samples data/nuscenes/samples
# ln -s /horizon-bucket/HCT_Perception2/public_dataset/vad_nuscenes/v1.0-mini data/nuscenes/v1.0-mini
# ln -s /horizon-bucket/HCT_Perception2/public_dataset/vad_nuscenes/v1.0-trainval data/nuscenes/v1.0-trainval

mkdir -p data/bench2drive/splits
ln -s /horizon-bucket/HCT_Perception2/public_dataset/Bench2Drive/v2 data/bench2drive/v1
ln -s /horizon-bucket/HCT_Perception2/public_dataset/Bench2Drive/maps data/bench2drive/maps
