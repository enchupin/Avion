@echo off
echo TSD-SR Super-Resolution 시작합니다...
echo.

python test/test_tsdsr.py ^
--pretrained_model_name_or_path checkpoint/tsdsr ^
-i imgs/test ^
-o outputs/test ^
--lora_dir checkpoint/tsdsr ^
--embedding_dir dataset/default/

echo.
echo 작업이 완료되었습니다. 결과는 outputs/test 폴더에서 확인하세요.
pause
