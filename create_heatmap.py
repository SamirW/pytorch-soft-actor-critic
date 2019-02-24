import argparse
import pickle
import matplotlib.pyplot as plt
from pathlib import Path
from algorithms.sac import SAC
from utils.heatmap import heatmap
from utils.heatmap3 import heatmap3
from utils.heatmap4 import heatmap4
from utils.buffer import ReplayBuffer

def run(config):
    model_dir = Path('./models') / config.env_id / config.model_name / "run{}".format(config.run)
    model_file = str(model_dir / "model.pt")

    sac = SAC.init_from_save(model_file)
    nagents = sac.nagents
    if nagents == 2:
        heatmap_fn = heatmap 
    elif nagents == 3:
        heatmap_fn = heatmap3
    else: # 4 agents
        heatmap_fn = heatmap4

    heatmap_fn(sac, title="{} Agent Policies Before Distillation".format(nagents), save=config.save, continuous=True)

    print("Distilling")
    with open(str(model_dir / "replay_buffer.pkl"), 'rb') as input:
        replay_buffer = pickle.load(input)
    sac.distill(2048, 1024, replay_buffer)

    print("Creating distilled heatmap")
    heatmap_fn(sac, title="{} Agent Policies After Distillation".format(nagents), save=config.save, continuous=True)

    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("env_id", help="Name of environment")
    parser.add_argument("model_name",
                        help="Name of directory to store " +
                             "model/training contents")
    parser.add_argument("run", help="Run number")
    parser.add_argument("--save",
                        action="store_true",
                        default=False)
    config = parser.parse_args()

    run(config)
