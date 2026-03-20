import pufferlib
from pufferlib.ocean.ctf import ctf
import numpy as np


def test_puffer_ctf():
    print("Testing Puffer CTF (C implementation)...")
    env = ctf.CTF(num_envs=1)
    obs, info = env.reset()
    print(f"Reset successful. Observation shape: {obs.shape}")

    for i in range(10):
        actions = np.random.randint(0, 17, size=4)
        obs, rewards, terminals, truncations, infos = env.step(actions)
        if i == 0:
            print(f"Step successful. Rewards: {rewards}")

    env.close()
    print("Puffer CTF test completed!")


if __name__ == "__main__":
    test_puffer_ctf()
