# Identify Tumor
# Unet++
python3 run_multigpu.py --encoder resnet18 --decoder UnetPlusPlus --batch_size 20
python3 run_multigpu.py --encoder resnet34 --decoder UnetPlusPlus --batch_size 18
python3 run_multigpu.py --encoder resnet50 --decoder UnetPlusPlus --batch_size 10
python3 run_multigpu.py --encoder efficientnet-b0 --decoder UnetPlusPlus --batch_size 20
python3 run_multigpu.py --encoder efficientnet-b1 --decoder UnetPlusPlus --batch_size 18
# Unet
python3 run_multigpu.py --encoder resnet18 --decoder Unet --batch_size 48
python3 run_multigpu.py --encoder resnet34 --decoder Unet --batch_size 42
python3 run_multigpu.py --encoder resnet50 --decoder Unet --batch_size 28
python3 run_multigpu.py --encoder efficientnet-b0 --decoder Unet --batch_size 28
python3 run_multigpu.py --encoder efficientnet-b1 --decoder Unet --batch_size 22