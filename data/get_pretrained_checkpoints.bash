# Get pretrained checkpoints
gdown https://drive.google.com/uc?id=1T3eltI8aHI4IUgX4yEunfYhZtVvLNoLX
unzip CenterLineDet_Checkpoints.zip
rm -rf CenterLineDet_Checkpoints.zip 

# mv checkpoints
mkdir -p ../CenterLineDet/FusionNet/checkpoints/
mv ./CenterLineDet_Checkpoints/center_fusionnet_best.pt ../CenterLineDet/FusionNet/checkpoints

mkdir mkdir -p ../CenterLineDet/HDMapNet/checkpoints/
mv ./CenterLineDet_Checkpoints/center_HDMapNet_fusion_best.pt ../CenterLineDet/HDMapNet/checkpoints

mkdir mkdir -p ../segmentation_baselines/FusionNet/checkpoints/
mv ./CenterLineDet_Checkpoints/fusionnet_best.pt ../segmentation_baselines/FusionNet/checkpoints

mkdir mkdir -p ../segmentation_baselines/HDMapNet/checkpoints/
mv ./CenterLineDet_Checkpoints/HDMapNet_fusion_best.pt ../segmentation_baselines/HDMapNet/checkpoints

rm -rf ./CenterLineDet_Checkpoints