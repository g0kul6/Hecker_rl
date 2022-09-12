python citylearn-2022-starter-kit/train.py --algo "ddpg" --actor_lr 3e-4 --critic_lr 3e-4 --gamma 0.99 --tau 0.05 --device "cuda" --epochs 1000 --reward_key 0
python citylearn-2022-starter-kit/train.py --algo "ddpg" --actor_lr 3e-4 --critic_lr 3e-4 --gamma 0.99 --tau 0.05 --device "cuda" --epochs 1000 --reward_key 1
python citylearn-2022-starter-kit/train.py --algo "ddpg" --actor_lr 3e-4 --critic_lr 3e-4 --gamma 0.99 --tau 0.05 --device "cuda" --epochs 1000 --reward_key 2


python citylearn-2022-starter-kit/drl_algo/main.py --cuda --policy "Gaussian" --reward_key 0
python citylearn-2022-starter-kit/drl_algo/main.py --cuda --policy "Gaussian" --reward_key 1
python citylearn-2022-starter-kit/drl_algo/main.py --cuda --policy "Gaussian" --reward_key 2


python citylearn-2022-starter-kit/drl_algo/main.py --cuda --policy "Deterministic" --reward_key 0
python citylearn-2022-starter-kit/drl_algo/main.py --cuda --policy "Deterministic" --reward_key 1
python citylearn-2022-starter-kit/drl_algo/main.py --cuda --policy "Deterministic" --reward_key 2


python citylearn-2022-starter-kit/train.py --algo "td3" --actor_lr 3e-4 --critic_lr 3e-4 --gamma 0.99 --tau 0.05 --device "cuda" --epochs 1000 --reward_key 0
python citylearn-2022-starter-kit/train.py --algo "td3" --actor_lr 3e-4 --critic_lr 3e-4 --gamma 0.99 --tau 0.05 --device "cuda" --epochs 1000 --reward_key 1
python citylearn-2022-starter-kit/train.py --algo "td3" --actor_lr 3e-4 --critic_lr 3e-4 --gamma 0.99 --tau 0.05 --device "cuda" --epochs 1000 --reward_key 2
