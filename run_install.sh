

#pip install numpy==1.18.2  pillow==6.2.1   && wget https://raw.githubusercontent.com/arita37/mlmodels/dev/requirements_fake.txt  && pip install -r requirements_fake.txt   && pip install  https://github.com/arita37/mlmodels/archive/dev.zip 



pip install numpy==1.18.2  pillow==6.2.1  



wget https://raw.githubusercontent.com/arita37/mlmodels/dev/requirements.txt  
pip install -r requirements.txt   
rm requirements.txt  ;


wget https://raw.githubusercontent.com/arita37/mlmodels/dev/requirements_fake.txt  
pip install -r requirements_fake.txt   
rm requirements_fake.txt  ;


### Extra for Colab
pip install torchvision==0.4.0



mkdir z  ;  cd z && git clone https://github.com/arita37/mlmodels.git  ;  


pip uninstall mlmodels ; pip install -e .  --no-deps  --force ;



python -c "import mlmodels;"










