import click

from mlmodels.util import log

@click.group()
def main():
    pass


### Shortcut ????
def add(*w, **kw):
        click.option(*w, **kw)


###############################################################################
################### mlmodels.py ###############################################
@main.command()
@click.option('-f', '--folder', help="Use option to mention folder explictly")
def list(folder=None):
    from mlmodels.mlmodels import config_model_list
    config_model_list(folder)


@main.command()
@click.option('-f', '--folder', help="Use option to mention folder explictly")
def testall():
    from mlmodels.mlmodels import test_all
    test_all(folder)



@main.command()
@click.argument('model_name')
@click.option('-p', '--params', help="Enter Model Params")
def test(model_name, params):
   from mlmodels.models import test_cli 
   test_cli(model_name, params)


@main.command()
@click.argument('model_name')
@click.option('-cf', '--config_file', help="Enter Path to config file")
@click.option('cm', '--config_mode', help="Enter Config Mode: test/prod/uat", default="test")
@click.option('-sf', '--save_folder', help="Model Save Location")
def fit(model_name, config_file, config_mode, save_folder):
    from mlmodels.models import fit_cli
    fit_cli(model_name, config_file, config_mode, save_folder)



@main.command()
@click.argument('model_name')
@click.option('-cf', '--config_file', help="Enter Path to config file")
@click.option('cm', '--config_mode', help="Enter Config Mode: test/prod/uat", default="test")
@click.option('-sf', '--save_folder', help="Model Save Location")
def predict():
    from mlmodels.models import predict_cli
    predict_cli(model_name, config_file, config_mode, save_folder)



@main.command()
@click.argument('model_name')
@click.option('-sf', '--save_folder', help="Folder Path to save configuration")
def generate_config(model_name, save_folder):
    from mlmodels.models import config_generate_json    
    log(arg.save_folder)
    config_generate_json(model_name, to_path=save_folder)



###############################################################################
################### optim.py ##################################################





###############################################################################
if __name__ == "__main__":
    main()