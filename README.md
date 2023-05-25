# optuna-mnist
A template example how to use optuna with pytorch lightning

```sh
python -m pip install --requirement requirements.txt
```

Start the mysql storage server and the optuna-dashboard

```sh
docker-compose up -d
```

Start the search

```sh
export OPTUNA_STORAGE=mysql+pymysql://root:start123@127.0.0.1:13306/optuna
./search.py
```

Start the tensorboard

```sh
tensorboard --logdir logs/
```