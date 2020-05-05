# -*- coding: utf-8 -*-
"""
Models are stored in model_XX/  or in folder XXXXX
    module :  folder/mymodel.py, contains the methods, operations.
    model  :  Class in mymodel.py containing the model definition, compilation
   
models.py   #### Generic Interface
   module_load(model_uri)
   model_create(module)
   fit(model, module, session, data_pars, out_pars   )
   metrics(model, module, session, data_pars, out_pars)
   predict(model, module, session, data_pars, out_pars)
   save(save_pars)
   load(load_pars)
 
######### Command line sample  #####################################################################
"""
import argparse
import glob
import inspect
import json
import os
import re
import sys
from importlib import import_module
from pathlib import Path
from warnings import simplefilter

####################################################################################################
from mlmodels.util import (get_recursive_files, load_config, log, os_package_root_path, path_norm)

from mlmodels.util import (env_build, env_conda_build, env_pip_requirement)

simplefilter(action='ignore', category=FutureWarning)
simplefilter(action='ignore', category=DeprecationWarning)


####################################################################################################
def module_env_build(model_uri="", verbose=0, do_env_build=0):
    """
      Load the file which contains the model description
      model_uri:  model_tf.1_lstm.py  or ABSOLUTE PATH
    """
    # print(os_file_current_path())
    model_uri = model_uri.replace("/", ".")
    module = None
    if verbose:
        print(model_uri)

    #### Dynamic ENV Build based on requirements.txt
    if do_env_build:
        env_pars = {"python_version": '3.6.5'}
        env_build(model_uri, env_pars)


def module_load(model_uri="", verbose=0, env_build=0):
    """
      Load the file which contains the model description
      model_uri:  model_tf.1_lstm.py  or ABSOLUTE PATH
    """
    # print(os_file_current_path())
    model_uri = model_uri.replace("/", ".")
    module = None
    if verbose:
        print(model_uri)

    try:
        #### Import from package mlmodels sub-folder
        model_name = model_uri.replace(".py", "")
        module = import_module(f"mlmodels.{model_name}")
        # module    = import_module("mlmodels.model_tf.1_lstm")

    except Exception as e1:
        try:
            ### Add Folder to Path and Load absoluate path model
            path_parent = str(Path(model_uri).parent.absolute())
            sys.path.append(path_parent)
            # print(path_parent, sys.path)

            #### import Absilute Path model_tf.1_lstm
            model_name = Path(model_uri).stem  # remove .py
            model_name = str(Path(model_uri).parts[-2]) + "." + str(model_name)
            # print(model_name)
            module = import_module(model_name)

        except Exception as e2:
            raise NameError(f"Module {model_name} notfound, {e1}, {e2}")

    if verbose: print(module)
    return module


def module_load_full(model_uri="", model_pars=None, data_pars=None, compute_pars=None, choice=None, **kwarg):
    """
      Create Instance of the model, module
      model_uri:  model_tf.1_lstm.py
    """
    module = module_load(model_uri=model_uri)
    model = module.Model(model_pars=model_pars, data_pars=data_pars, compute_pars=compute_pars, **kwarg)
    return module, model


def model_create(module, model_pars=None, data_pars=None, compute_pars=None, **kwarg):
    """
      Create Instance of the model from loaded module
      model_pars : dict params
    """
    if model_pars is None:
        model_pars = module.get_params()

    model = module.Model(model_pars=model_pars, data_pars=data_pars, compute_pars=compute_pars, **kwarg)
    return model


def fit(module, model, sess=None, data_pars=None, compute_pars=None, out_pars=None, **kwarg):
    """
    Wrap fit generic method
    :type model: object
    """

    # module, model = module_load_full(model_uri, model_pars, data_pars, compute_pars)
    # sess=None
    return module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars, **kwarg)


def predict(module, model, sess=None, data_pars=None, compute_pars=None, out_pars=None, **kwarg):
    """
       predict  using a pre-trained model and some data
    :return:
    """
    # module      = module_load(model_uri)
    # model,sess  = load(model_pars)

    return module.predict(model, sess, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars, **kwarg)


def fit_metrics(module, model, sess=None, data_pars=None, compute_pars=None, out_pars=None, **kwarg):
    return module.fit_metrics(model, sess, data_pars, compute_pars, out_pars, **kwarg)


def get_params(module, params_pars, **kwarg):
    return module.get_params(params_pars, **kwarg)


def metrics(module, model, sess=None, data_pars=None, compute_pars=None, out_pars=None, **kwarg):
    return module.metrics(model, sess, data_pars, compute_pars, out_pars, **kwarg)


def load(module, load_pars, **kwarg):
    """
       Load model/session from files
       :param folder_name:
    """
    return module.load(load_pars, **kwarg)


def save(module, model, session, save_pars, **kwarg):
    """
       Save model/session on disk
    """
    return module.save(model, session, save_pars, **kwarg)


####################################################################################################
####################################################################################################
def test_all(folder=None):
    if folder is None:
        folder = os_package_root_path() + "/model_tf/"

    # module_names = get_recursive_files(folder, r"[0-9]+_.+\.py$")
    module_names = config_model_list()
    module_names.sort()
    print(module_names)
    failed_scripts = []

    for module_name in module_names:
        print("#######################")
        print(module_name)
        test(module_name)


def test(modelname):
    print(modelname)
    try:
        module = module_load(modelname, verbose=1)
        print(module)
        module.test()
        del module
    except Exception as e:
        print("Failed", e)


def test_global(modelname):
    print(modelname)
    try:
        module = module_load(modelname, verbose=1)
        print(module)
        module.test()
        del module
    except Exception as e:
        print("Failed", e)





def test_api(model_uri="model_xxxx/yyyy.py", param_pars=None):
    log("############ Model preparation   ##################################")
    from mlmodels.models import module_load_full
    from mlmodels.models import fit as fit_global
    from mlmodels.models import predict as predict_global
    from mlmodels.models import save as save_global, load as load_global

    log("#### Module init   ############################################")
    from mlmodels.models import module_load
    module = module_load(model_uri)
    log(module)

    log("#### Loading params   ##############################################")
    model_pars, data_pars, compute_pars, out_pars = get_params(module, param_pars)

    log("#### Model init   ############################################")
    session = None
    # model_sklearn change model_pars => could not init model in sequence or need get_params before
    # from mlmodels.models import model_create
    # model = model_create(module, model_pars, data_pars, compute_pars)

    module, model = module_load_full(model_uri, model_pars, data_pars, compute_pars)

    log("############ Model fit   ##########################################")
    model, sess = fit_global(module, model, sess=None, data_pars=data_pars, compute_pars=compute_pars,
                             out_pars=out_pars)
    print("fit success", sess)

    log("############ Prediction############################################")
    ### Load model, and predict 
    preds = predict_global(module, model, session, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars)
    print(preds)

    log("############ Save/ Load ############################################")
    # save_global( save_pars, model, sess)
    # load_global(save_pars)


def test_module(model_uri="model_xxxx/yyyy.py", param_pars=None):
    # Using local method only

    log("#### Module init   ############################################")
    from mlmodels.models import module_load
    module = module_load(model_uri)
    log(module)

    log("#### Loading params   ##############################################")
    # param_pars = {"choice":pars_choice,  "data_path":data_path,  "config_mode": config_mode}
    model_pars, data_pars, compute_pars, out_pars = module.get_params(param_pars)

    log("#### Model init   ############################################")
    model = module.Model(model_pars, data_pars, compute_pars)
    log(model)

    log("#### Fit   ########################################################")
    model, sess = module.fit(model, data_pars, compute_pars, out_pars)

    log("#### Predict   ####################################################")
    ypred = module.predict(model, sess, data_pars, compute_pars, out_pars)
    print(ypred)

    log("#### Get  metrics   ################################################")
    metrics_val = module.fit_metrics(model, data_pars, compute_pars, out_pars)

    log("#### Save   ########################################################")
    # save_pars = {}
    # load_pars = {}
    # module.save( save_pars,  model, sess)

    log("#### Load   ########################################################")
    # model2, sess2 = module.load(load_pars)
    #     ypred = predict(model2, data_pars, compute_pars, out_pars)
    #     metrics_val = metrics(model2, ypred, data_pars, compute_pars, out_pars)
    # print(model2)


####################################################################################################
############ JSON template #########################################################################
def config_get_pars(config_file, config_mode="test"):
    """
      load JSON and output the params
    """
    js = json.load(open(config_file, 'r'))  # Config
    js = js[config_mode]  # test /uat /prod
    model_p = js.get("model_pars")
    data_p = js.get("data_pars")
    compute_p = js.get("compute_pars")
    out_p = js.get("out_pars")

    return model_p, data_p, compute_p, out_p


def config_generate_json(modelname, to_path="ztest/new_model/"):
    """
      Generate config file from code source
      config_init("model_tf.1_lstm", to_folder="ztest/")
    """
    os.makedirs(to_path, exist_ok=True)
    ##### JSON file
    import inspect
    module = module_load(modelname)
    signature = inspect.signature(module.Model)
    args = {
        k: v.default if v.default is not inspect.Parameter.empty else None
        for k, v in signature.parameters.items()
        # if v.default is not inspect.Parameter.empty
    }

    # args = inspect.getargspec(module.Model)
    model_pars = {"model_pars": args,
                  "data_pars": {},
                  "compute_pars": {},
                  "out_pars": {}
                  }

    modelname = modelname.replace(".py", "").replace(".", "-")
    fname = os.path.join(to_path, f"{modelname}_config.json")
    json.dump(model_pars, open(fname, mode="w"))
    print(fname)


def os_folder_copy(src, dst):
    """Copy a directory structure overwriting existing files"""
    import shutil
    for root, dirs, files in os.walk(src):
        if not os.path.isdir(root):
            os.makedirs(root, exist_ok=True)

        for file in files:
            rel_path = root.replace(src, '').lstrip(os.sep)
            dest_path = os.path.join(dst, rel_path)

            if not os.path.isdir(dest_path):
                os.makedirs(dest_path, exist_ok=True)

            try:
                shutil.copyfile(os.path.join(root, file), os.path.join(dest_path, file))
            except Exception as e:
                print(e)


def config_init(to_path="."):
    """
      Generate template from code source
      config_init("model_tf.1_lstm", to_folder="ztest/")
    """
    import shutil
    os_root = os_package_root_path()

    to_path = os_root + "/ztest/current/" if to_path == "." else to_path
    log("Working Folder", to_path)
    # os.makedirs(to_path, exist_ok=True)

    os_folder_copy(os_root + "/template/", to_path + "/template/")
    os_folder_copy(os_root + "/dataset/", to_path + "/dataset/")
    os_folder_copy(os_root + "/example/", to_path + "/example/")

    os.makedirs(to_path + "model_trained", exist_ok=True)
    os.makedirs(to_path + "model_code", exist_ok=True)

    #### Config files
    path_user = os.path.expanduser('~')
    path_config = path_user + "/.mlmodels/config.json"
    # print("config file", path_config)

    os.makedirs(path_user + "/.mlmodels/", exist_ok=True)
    ddict = {"model_trained": to_path + "/model_trained/",
             "dataset": to_path + "/dataset/", }
    log("Config values", ddict)
    json.dump(ddict, open(path_config, mode="w"))

    from mlmodels.util import config_path_pretrained, config_path_dataset
    log("Config path", get_pretrained_path())


def config_model_list(folder=None):
    # Get all the model.py into folder
    folder = os_package_root_path() if folder is None else folder
    print(folder)
    module_names = get_recursive_files(folder, r'/*model*/*.py')
    mlist = []
    for t in module_names:
        mlist.append(t.replace(folder, "").replace("\\", "."))
        print(mlist[-1])

    return mlist



####################################################################################################
############CLI Command ############################################################################
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

    add("--config_file", default=config_file, help="Params File")
    add("--config_mode", default="test", help="test/ prod /uat")
    add("--log_file", default="mlmodels_log.log", help="log.log")
    add("--do", default="test", help="do ")
    add("--folder", default=None, help="folder ")

    add("--init", default="", help=".")

    ##### model pars
    add("--model_uri", default="model_tf/1_lstm.py", help=".")
    add("--load_folder", default="ztest/", help=".")

    ##### data pars
    add("--dataname", default="dataset/google.csv", help=".")

    ##### compute pars

    ##### out pars
    add("--save_folder", default="ztest/", help=".")

    arg = p.parse_args()
    # arg = load_config(arg, arg.config_file, arg.config_mode, verbose=0)
    return arg


def main():
    arg = cli_load_arguments()
    print(arg.do)

    if len(arg.init) > 0:
        config_init(to_path=arg.init)
        return 0

    if arg.do == "generate_config":
        log(arg.save_folder)
        config_generate_json(arg.model_uri, to_path=arg.save_folder)

    ###################################################################
    if arg.do == "model_list":  # list all models in the repo
        l = config_model_list(arg.folder)

    if arg.do == "testall":
        # test_all() # tot test all te modules inside model_tf
        test_all(folder=None)

    if arg.do == "test":
        param_pars = {"choice": "test01", "data_path": "", "config_mode": "test"}
        test_module(arg.model_uri, param_pars=param_pars)  # '1_lstm'

        test(arg.model_uri)  # '1_lstm'
        # test_api(arg.model_uri)  # '1_lstm'
        test_global(arg.model_uri)  # '1_lstm'

    if arg.do == "fit":
        model_p, data_p, compute_p, out_p = config_get_pars(arg.config_file, arg.config_mode)

        module = module_load(arg.model_uri)  # '1_lstm.py
        model = model_create(module, model_p, data_p, compute_p)  # Exact map JSON and paramters

        log("Fit")
        model, sess = module.fit(model, data_pars=data_p, compute_pars=compute_p, out_pars=out_p)

        log("Save")
        save_pars = {"path": f"{arg.save_folder}/{arg.model_uri}", "model_uri": arg.model_uri}
        save(save_pars, model, sess)

    if arg.do == "predict":
        model_p, data_p, compute_p, out_p = config_get_pars(arg.config_file, arg.config_mode)
        # module = module_load(arg.modelname)  # '1_lstm'
        load_pars = {"path": f"{arg.save_folder}/{arg.model_uri}", "model_uri": arg.model_uri}

        module = module_load(model_p[".model_uri"])  # '1_lstm.py
        model, session = load(load_pars)
        module.predict(model, session, data_pars=data_p, compute_pars=compute_p, out_pars=out_p)


if __name__ == "__main__":
    main()







"""
def main():
    cli = argparse.ArgumentParser()
    subparsers = cli.add_subparsers(dest="cli_")

    def add(*names_or_flags, **kwargs):
            return names_or_flags, kwargs


    def cli_(*subparser_args, parent=subparsers):
            def decorator(func):
                parser = parent.add_parser(func.__name__, description=func.__doc__)
                for args, kwargs in subparser_args:
                    parser.add_add(*args, **kwargs)
                parser.set_defaults(func=func)
            return decorator



    @cli_( add("--path", help="Enter path to create workspace", default="."))
    def init(args):
        config_init(to_path=args.path)
        return 0



    @cli_(
        add("--save_folder", help="Location to save the generated configuration", default="ztest/"),
        add("--model_uri", default="model_tf/1_lstm.py", help="Model Name")
    )
    def generate_config(args):
        log(args.save_folder)
        config_generate_json(args.model_uri, to_path=args.save_folder)




    @cli_( add("--folder", help="Enter the path with all models", default=None) )
    def model_list(args):
        l = config_model_list(args.folder)



    @cli_( add("--folder", help="Enter the path with all models", default=None))
    def testall(args):
        test_all(folder=args.folder)



    @cli_( add("--model_uri", help="Enter Model URI", default="model_tf/1_lstm.py"))
    def test(args):
        param_pars = {"choice": "test01", "data_path": "", "config_mode": "test"}
        test_module(args.model_uri, param_pars=param_pars)  # '1_lstm'
        test(args.model_uri)
        test_global(args.model_uri)



    @cli_(
        add("--config_file", help="Path to config file", default=None),
        add("--config_mode", help="test/ prod /uat", default="test"),
        add("--model_uri", help="Enter Model URI", default="model_tf/1_lstm.py"),
        add("--save_folder", help="Location to save the generated configuration", default="ztest/")
    )
    def fit(args):
        if args.config_file is None:
            cur_path = os.path.dirname(os.path.realpath(__file__))
            args.config_file = os.path.join(cur_path, "template/models_config.json")
        model_p, data_p, compute_p, out_p = config_get_pars(args.config_file, args.config_mode)
        module = module_load(args.model_uri)  # '1_lstm.py
        model = model_create(module, model_p, data_p, compute_p)  # Exact map JSON and paramters
        log("Fit")
        model, sess = module.fit(model, data_pars=data_p, compute_pars=compute_p, out_pars=out_p)
        log("Save")
        save_pars = {"path": f"{args.save_folder}/{args.model_uri}", "model_uri": args.model_uri}
        save(save_pars, model, sess)




    @cli_(
        add("--config_file", help="Path to config file", default=None),
        add("--config_mode", help="test/ prod /uat", default="test"),
        add("--model_uri", help="Enter Model URI", default="model_tf/1_lstm.py"),
        add("--save_folder", help="Location to save the generated configuration", default="ztest/")
    )
    def predict(args):
        if args.config_file is None:
            args.config_file = path_norm("template/models_config.json")
            
        model_p, data_p, compute_p, out_p = config_get_pars(args.config_file, args.config_mode)
        # module = module_load(arg.modelname)  # '1_lstm'
        load_pars = {"path": f"{args.save_folder}/{args.model_uri}", "model_uri": args.model_uri}

        module = module_load(model_p[".model_uri"])  # '1_lstm.py
        model, session = load(load_pars)
        module.predict(model, session, data_pars=data_p, compute_pars=compute_p, out_pars=out_p)
    


    args = cli.parse_args()
    if args.cli_ is None:
        cli.print_help()
    else:
        args.func(args)

#def main():
#    cli()

"""
