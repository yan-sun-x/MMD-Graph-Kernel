from mmdgk.kernels import MMDGK, deep_MMDGK
from mmdgk.utils import arg_parse

if __name__ == '__main__':

    args = arg_parse()

    if args.model == "vanilla":
        kernel = MMDGK(args)
    else:
        kernel = deep_MMDGK(args)