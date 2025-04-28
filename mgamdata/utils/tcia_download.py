import os
import pdb
import argparse
from tqdm import tqdm

import pandas as pd

from tcia_utils import nbia


def download_meta(manifest_path:str, output_dir:str, retry:int):
    if os.path.exists(os.path.join(output_dir, "metadata.csv")):
        metas = pd.read_csv(os.path.join(output_dir, "metadata.csv"))
    else:
        metas = pd.DataFrame()
    
    series_uids = nbia.manifestToList(manifest_path)
    for series_uid in tqdm(series_uids, leave=False, dynamic_ncols=True):
        if series_uid in metas["Series UID"].values:
            continue
        
        retry_now = retry
        while retry_now:
            meta:list|None = nbia.getSeriesMetadata(series_uid)
            if meta is not None:
                break
            retry_now -= 1
        
        metas.append(pd.DataFrame(meta[0]))  # type: ignore

    output_path = os.path.join(output_dir, "metadata.csv")
    metas.to_csv(output_path, index=False)
    print(f"Downloaded metadata for {len(metas)} series, writting to {output_path}")
    
    return metas

def update_meta_filepath(data_root:str, meta_path:str):
    df = pd.read_csv(meta_path)
    
    for i in range(len(df)):
        series_uid = df.loc[i, "Series UID"]
        filepath = df.loc[i, "File Location"]
        if not os.path.exists(os.path.join(data_root, str(filepath))):
            print(f"Series {series_uid} not found according to meta file path.")
            if os.path.exists(os.path.join(data_root, str(series_uid))):
                print(f"auto found {series_uid} in {data_root}")
                df.loc[i, "File Location"] = series_uid
    
    df.to_csv(meta_path, index=False)
    return df
        
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest_path", type=str)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--retry", type=int, default=50)
    parser.add_argument("--only-meta", action="store_true", default=False)
    args = parser.parse_args()
    if args.output_dir is None:
        args.output_dir = os.path.join(os.path.dirname(args.manifest_path), "data")
    return args


if __name__ == "__main__":
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    retry = args.retry
    
    download_meta(args.manifest_path, args.output_dir, retry)
    
    if not args.only_meta:
        while retry:
            download_result = nbia.downloadSeries(
                series_data=args.manifest_path, 
                path=args.output_dir,
                input_type="manifest",
                csv_filename="metadata.csv",
                format="csv",
                as_zip=False,
            )
            
            if download_result is not None:
                break
            
            retry -= 1

    update_meta_filepath(args.output_dir, os.path.join(args.output_dir, "metadata.csv"))