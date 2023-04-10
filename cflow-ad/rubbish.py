import json

input_args = "--gpu 0 -enc wide_resnet50_2 --pro -inp 256 --dataset mvtec --class-name cable"

vscode_args = list(input_args.split(" "))
print(json.dumps(vscode_args))