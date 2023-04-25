import json

input_args = "--gpu 0 -enc wide_resnet50_2 --pro -inp 256 --dataset mvtec --class-name cable"

vscode_args = list(input_args.split(" "))
print(json.dumps(vscode_args))




def test_func(a, **kwargs):
    if 'b' in kwargs:
        b = kwargs['b']
        print(b)
    print(a)
    return


test_func(1, b=3)
test_func('xx',c=2)