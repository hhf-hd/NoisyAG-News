# WN    

nohup python -u main.py config/WN.yaml > logs/WN/log.log 2>&1 & 

# LS

nohup python -u main.py config/LS.yaml > logs/LS/log.log 2>&1 & 

# NLS

nohup python -u main.py config/NLS.yaml > logs/NLS/log.log 2>&1 & 

# CT

nohup python -u main.py config/CT.yaml > logs/CT/log.log 2>&1 & 

# selfMix

nohup python -u main.py config/selfMix.yaml > logs/selfMix/log.log 2>&1 & 

# DenoMix

nohup python -u main.py config/DenoMix.yaml > logs/DenoMix/log.log 2>&1 & 

# expDecay

nohup python -u main.py config/expDecay.yaml > logs/expDecay/log.log 2>&1 & 


