from train import train_DQN
from DQN_check_solution_full import check_solution


def main(args):
    # assign device
    if args.cuda:
        device = 'cuda:0'
    else:
        device = 'cpu'

    # perform training
    if args.train:
        # run training and plot avarage reward graphs
        train_DQN(device)

    # check solution of an existing checkpoint
    check_solution(args.path, device)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='')

    # Minotaur implementation options
    parser.add_argument('--train', action='store_true', help='Train DQN')
    parser.add_argument('--path', type=str, default='neural-network-1.pth',
                        help='Path to checkpoint. Plots Q graphs and checks solution')
    parser.add_argument('--cuda', action='store_true', help='Enable CUDA training/checking')

    args = parser.parse_args()

    main(args)
