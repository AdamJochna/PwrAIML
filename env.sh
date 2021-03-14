p_path=/home/adam/Desktop/PwrAIML

# SYSTEM PACKAGES UPDATE
sudo apt-get -y update
sudo apt-get -y upgrade
sudo apt-get -y dist-upgrade
sudo apt-get -y install python3.8
sudo update-alternatives --remove-all python3
sudo update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.8 1
sudo rm /usr/bin/python3
sudo ln -s python3.8 /usr/bin/python3
sudo apt-get -y install mysql-server libmysqlclient-dev
sudo apt-get -y install python3.8-venv python3.8-dev python3-setuptools
sudo apt-get -y install python3-pip python3-wheel
sudo apt-get -y install python-dev python-pip
sudo apt-get -y install gcc libpq-dev
sudo apt-get -y install libxml2-dev libxslt1-dev
sudo apt-get -y install python3-lxml


# VENV CREATION
python3.8 -m venv $p_path/venv
$p_path/venv/bin/python3 -m pip install -r $p_path/requirements.txt
