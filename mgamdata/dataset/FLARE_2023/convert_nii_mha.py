from ..base_convert import format_from_nnUNet

if __name__ == "__main__":
    formatter = format_from_nnUNet.start_from_argparse()
    formatter.execute()
