"""

Parse Code source CLI to extract JSON file

   


"""
import json
import  os, argparse
from pathlib import Path


def cli_load_arguments(config_file=None):
    """
        Load CLI input, load config.toml , overwrite config.toml by CLI Input
    """
    if config_file is None:
        cur_path = os.path.dirname(os.path.realpath(__file__))
        config_file = os.path.join(cur_path, "template/models_config.json")
    # print(config_file)

    p = argparse.ArgumentParser()

    def add(*w, **kw):
        p.add_argument(*w, **kw)

    add("--fromfile", default="model_tch/raw/vae_pretraining_encoder/text_beta.py", help="Params File")
    add("--tofile", default="config/json/model_tch/vae_pretraining_encoder.json", help="test/ prod /uat")


    arg = p.parse_args()
    # arg = load_config(arg, arg.config_file, arg.config_mode, verbose=0)
    return arg


def extract_args(txt, outfile):
    ddict ={}
    for ll in txt :
       # print(ll)
       if "add_argument(" in ll :
            ll = ll.replace("parser.argument(", "")
            ls = ll.split(",")


            for t in ls :
              if "default=" in t :
                val = t.replace("default=", "").replace(")", "").strip()
                key = ls[0].replace("parser.add_argument(", "").replace("--", "").replace("'", "").strip()
                ddict[key] = val

    def tonum(x):
      try :
         if "." in x :
           return float(x)
         else :
           return int(x)  
      except :
         return x

    ddict = { k : tonum(x) for k,x in ddict.items() }
    print(ddict)

    os.makedirs(  Path(outfile).parent, exist_ok=True)
    json.dump(ddict, open(outfile, mode="w"))




if __name__ == "__main__":
  arg = cli_load_arguments()

  file = arg.fromfile

  from mlmodels.util import get_recursive_files


  filelist = [file]
  for file in filelist :

    with open(file, mode="r") as f :
      txtjoin = f.read()

    if "argparse.ArgumentParser" in txtjoin :
       txt = txtjoin.split("\n")
       outfile = "config/json/" + file.replace(".py", ".json")
       extract_args(txt, outfile)







