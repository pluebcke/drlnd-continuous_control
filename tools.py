import csv
import glob
import os
import matplotlib.pyplot as plt

def save_results(base_path, agent, settings, rewards, average_rewards, a_loss, c_a_loss, c_b_loss):

    # Create directory for next results
    prefix = "run"
    existing_dirs = glob.glob(base_path + prefix + "*")
    suffix = str(len(existing_dirs)).zfill(3)

    save_path = base_path + prefix + suffix + "/"
    try:
        os.mkdir(save_path)
    except OSError:
        print("Path already exists " + save_path + " no results saved")
        return
    # plot results
    plt.figure(1)
    plt.plot(rewards)
    plt.plot(average_rewards)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt_path = save_path + "learning_curve.png"
    plt.savefig(plt_path, dpi=150)

    plt.figure(2)
    plt.plot(a_loss)
    plt.ylabel('Actor loss')
    plt.xlabel('Episode #')
    plt_path = save_path + "a_loss.png"
    plt.savefig(plt_path, dpi=150)

    plt.figure(3)
    plt.plot(c_a_loss)
    plt.plot(c_b_loss)
    plt.ylabel('Critic loss')
    plt.xlabel('Episode #')
    plt_path = save_path + "c_loss.png"
    plt.savefig(plt_path, dpi=150)

    # Save the agent neural networks
    agent.save_nets(save_path)

    # save the settings
    file = open(save_path + 'settings.csv', 'w')
    writer = csv.DictWriter(file, fieldnames=['key', 'value'])
    writer.writeheader()
    for key in settings.keys():
        writer.writerow({'key': key, 'value': settings[key]})
    file.close()
